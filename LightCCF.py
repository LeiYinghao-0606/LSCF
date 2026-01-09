import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

import utility.trainer as trainer
import utility.tools as tools
import utility.losses as losses


class LightCCF(nn.Module):
    def __init__(self, args, dataset, device):
        super(LightCCF, self).__init__()
        self.model_name = "LightCCF"
        self.dataset = dataset
        self.args = args
        self.device = device

        # ===== 可选：TF32 加速（主要加速 NA 里的大 matmul；不改目标函数）
        # 默认开启（allow_tf32=1），如果你担心数值差异可设为 0
        self.allow_tf32 = int(getattr(self.args, "allow_tf32", 1)) == 1
        if self.allow_tf32 and torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass

        # ===== 超参 =====
        self.reg_lambda = float(self.args.reg_lambda)
        self.activation = nn.Sigmoid()
        self.ssl_lambda = float(self.args.ssl_lambda)
        self.tau = float(self.args.tau)
        self.encoder = self.args.encoder  
        self.sg_alpha = float(getattr(self.args, "sg_alpha", 0.0))

        # ===== Embedding =====
        self.user_embedding = nn.Embedding(
            num_embeddings=self.dataset.num_users,
            embedding_dim=int(self.args.embedding_size)
        )
        self.item_embedding = nn.Embedding(
            num_embeddings=self.dataset.num_items,
            embedding_dim=int(self.args.embedding_size)
        )
        nn.init.xavier_uniform_(self.user_embedding.weight, gain=1.0)
        nn.init.xavier_uniform_(self.item_embedding.weight, gain=1.0)

        # ===== 邻接矩阵（用于 GCN 聚合）=====
        sp_adj = self.dataset.sparse_adjacency_matrix()  # scipy.sparse
        self.adj_mat = tools.convert_sp_mat_to_sp_tensor(sp_adj).to(self.device).coalesce()

        # 如果你明确想试 CSR（可能更快也可能更慢），把 args.spm_layout 设为 'csr'
        self.spm_layout = str(getattr(self.args, "spm_layout", "coo")).lower()
        if self.spm_layout == "csr":
            try:
                self.adj_mat = self.adj_mat.to_sparse_csr()
            except Exception:
                self.adj_mat = self.adj_mat.coalesce()

        # ===== 方案A：在模型初始化阶段一次性构造 item_pop，并缓存为 GPU Tensor =====
        self.item_pop = None  # torch.Tensor on device or None

        # 1) 如果 dataset 已经有 item_pop 且非空：直接用
        raw_pop = getattr(self.dataset, "item_pop", None)
        # 2) 否则：尝试用 dataset.train_items 统计得到
        if raw_pop is None:
            train_items = getattr(self.dataset, "train_items", None)
            if train_items is not None:
                if torch.is_tensor(train_items):
                    train_items_np = train_items.detach().cpu().numpy().astype(np.int64, copy=False)
                else:
                    train_items_np = np.asarray(train_items, dtype=np.int64)

                # bincount 要求非负整数 id
                pop_np = np.bincount(train_items_np, minlength=self.dataset.num_items).astype(np.float32, copy=False)

                # 写回 dataset，后续别的模块也能复用
                self.dataset.item_pop = pop_np
                raw_pop = pop_np
        # 3) 缓存到 GPU（只做一次，避免 forward 里反复搬运）
        if raw_pop is not None:
            if torch.is_tensor(raw_pop):
                self.item_pop = raw_pop.to(self.device, non_blocking=True).float()
            else:
                self.item_pop = torch.as_tensor(raw_pop, device=self.device, dtype=torch.float32)

    def aggregate(self):
        embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [embeddings]

        for layer in range(int(self.args.gcn_layer)):
            embeddings = torch.sparse.mm(self.adj_mat, embeddings)
            all_embeddings.append(embeddings)
        final_embeddings = torch.stack(all_embeddings, dim=1)
        final_embeddings = torch.mean(final_embeddings, dim=1)

        user_emb, item_emb = torch.split(final_embeddings, [self.dataset.num_users, self.dataset.num_items])

        return user_emb, item_emb

    def forward(self, user, positive, negative):
        # ===== 1. GCN / encoder 表示（用于 NA & 结构融合）=====
        if self.encoder == "MF":
            all_user_gcn_embed = self.user_embedding.weight
            all_item_gcn_embed = self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        user_gcn_embed = all_user_gcn_embed[user.long()]          # [B, d]
        positive_gcn_embed = all_item_gcn_embed[positive.long()]  # [B, d]
        negative_gcn_embed = all_item_gcn_embed[negative.long()]  # [B, d]

        # ===== 2. MF 表示 =====
        user_mf_embed = self.user_embedding(user)                 # [B, d]
        positive_mf_embed = self.item_embedding(positive)         # [B, d]
        negative_mf_embed = self.item_embedding(negative)         # [B, d]

        # ===== 2.1 Stop-Gradient 结构融合：构造用于 BPR 的混合表示 =====
        if self.sg_alpha > 0.0 and self.encoder != "MF":
            alpha = self.sg_alpha
            user_for_bpr = user_mf_embed + alpha * (user_gcn_embed.detach() - user_mf_embed.detach())
            pos_for_bpr = positive_mf_embed + alpha * (positive_gcn_embed.detach() - positive_mf_embed.detach())
            neg_for_bpr = negative_mf_embed + alpha * (negative_gcn_embed.detach() - negative_gcn_embed.detach())
            # ↑ 注意：这里原来你写的是 (neg_gcn - neg_mf)，我保留正确形式如下
            neg_for_bpr = negative_mf_embed + alpha * (negative_gcn_embed.detach() - negative_mf_embed.detach())
        else:
            user_for_bpr = user_mf_embed
            pos_for_bpr = positive_mf_embed
            neg_for_bpr = negative_mf_embed

        # ===== 3. BPR =====
        bpr_loss = losses.get_bpr_loss(user_for_bpr, pos_for_bpr, neg_for_bpr)

        # ===== 4. L2 正则 =====
        reg_loss = losses.get_reg_loss(
            user_mf_embed, positive_mf_embed, negative_mf_embed
        ) * self.reg_lambda

        # ===== 5. NA（重要性采样 / 均匀采样：由 sample_weight 是否为 None 决定）=====
        neg_k = int(getattr(self.args, "na_neg_k", 8192))
        beta = float(getattr(self.args, "na_q_beta", 0.5))
        mix_u = float(getattr(self.args, "na_mix_uniform", 0.7))

        sample_weight = None
        if self.item_pop is not None:
            # 用 batch 的正样本 item 热度作为先验权重
            w = self.item_pop[positive.long()].clamp_min(1.0)  # [B]
            sample_weight = torch.pow(w, beta)

        na_loss = losses.get_neighbor_aggregate_loss(
            user_gcn_embed, positive_gcn_embed, self.tau,
            neg_k=neg_k,
            sample_weight=sample_weight,
            mix_uniform=mix_u
        ) * self.ssl_lambda

        loss_list = [bpr_loss, reg_loss, na_loss]


        return loss_list

    def get_rating_for_test(self, user):
        """
        测试阶段：
        - 若启用 SGSF（sg_alpha > 0 且 encoder != 'MF'），则用混合后的表示打分；
        - 否则保持原行为：GCN 表示（或纯 MF）。
        """
        if self.encoder == "MF":
            all_user_gcn_embed = self.user_embedding.weight
            all_item_gcn_embed = self.item_embedding.weight
        else:
            all_user_gcn_embed, all_item_gcn_embed = self.aggregate()

        all_user_mf_embed = self.user_embedding.weight
        all_item_mf_embed = self.item_embedding.weight

        if self.sg_alpha > 0.0 and self.encoder != "MF":
            alpha = self.sg_alpha
            all_user_final = all_user_mf_embed + alpha * (all_user_gcn_embed - all_user_mf_embed)
            all_item_final = all_item_mf_embed + alpha * (all_item_gcn_embed - all_item_mf_embed)
        else:
            if self.encoder == "MF":
                all_user_final = all_user_mf_embed
                all_item_final = all_item_mf_embed
            else:
                all_user_final = all_user_gcn_embed
                all_item_final = all_item_gcn_embed

        user_embed = all_user_final[user.long()]   # [B, d]
        rating = self.activation(torch.matmul(user_embed, all_item_final.t()))
        return rating


class Trainer:
    def __init__(self, args, dataset, device, logger):
        self.model = LightCCF(args, dataset, device)
        self.args = args
        self.dataset = dataset
        self.device = device
        self.logger = logger

    def train(self):
        trainer.training(self.model, self.args, self.dataset, self.device, self.logger)
