import torch
import torch.nn.functional as F
import numpy as np
import math

def get_bpr_loss(user_embed, pos_embed, neg_embed):
    pos_scores = torch.sum(torch.mul(user_embed, pos_embed), dim=1)
    neg_scores = torch.sum(torch.mul(user_embed, neg_embed), dim=1)
    loss = -torch.log(torch.sigmoid(pos_scores - neg_scores) + 10e-8)
    return torch.mean(loss)


def get_focal_bpr_loss(user_embed, pos_embed, neg_embed, gamma=0.0):
    pos_scores = torch.sum(torch.mul(user_embed, pos_embed), dim=1)
    neg_scores = torch.sum(torch.mul(user_embed, neg_embed), dim=1)
    margin = pos_scores - neg_scores
    prob = torch.sigmoid(margin)
    weight = (1.0 - prob).pow(gamma)
    loss = -weight * torch.log(prob + 10e-8)
    return torch.mean(loss)


def get_reg_loss(*embeddings):
    reg_loss = 0.0
    for emb in embeddings:
        reg_loss += torch.sum(emb ** 2) / emb.shape[0]
    return reg_loss


def get_cl_loss(embedding1, embedding2, temperature):
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    pos_score = (embedding1 * embedding2).sum(dim=-1)
    pos_score = torch.exp(pos_score / temperature)

    total_score = torch.matmul(embedding1, embedding2.transpose(0, 1))
    total_score = torch.exp(total_score / temperature).sum(dim=1)

    cl_loss = -torch.log(pos_score / total_score + 10e-6)
    return torch.mean(cl_loss)


def get_neighbor_aggregate_loss(embedding1, embedding2, tau,
                               neg_k: int = 0,
                               sample_weight: torch.Tensor = None,
                               mix_uniform: float = 0.5,
                               eps: float = 1e-12):
    """
    Neighbor-Aggregate loss with two modes:

    (A) Exact (default): O(B*B)
        denom = sum_j exp( u_i · (v_j + u_j) / tau )

    (B) Importance-sampled: O(B*K)
        Z_i is estimated by Monte-Carlo importance sampling:
            Z_hat = (1/K) * sum_{k} exp(score_{s_k}) / q(s_k), s_k ~ q
        where q is a mixture of uniform and data prior weights.

    Args:
        embedding1: [B, d]
        embedding2: [B, d]
        tau: temperature
        neg_k: number of sampled terms K (0 or >=B -> exact)
        sample_weight: [B] nonnegative weights from data prior (e.g., popularity^beta)
        mix_uniform: lambda in q = lambda * uniform + (1-lambda) * normalized(sample_weight)
    """
    u = F.normalize(embedding1, dim=-1)
    v = F.normalize(embedding2, dim=-1)

    # numerator: exp((u_i · v_i)/tau)
    pos_logit = torch.sum(u * v, dim=1) / tau  # [B]

    B = u.size(0)
    cand = v + u  # [B, d]  (exact algebra: u·v_j + u·u_j = u·(v_j+u_j))

    # -------- Exact path (fastest when you can afford BxB) --------
    if (neg_k is None) or (neg_k <= 0) or (neg_k >= B):
        logits = torch.matmul(u, cand.t()) / tau  # [B, B]
        log_denom = torch.logsumexp(logits, dim=1)  # [B]
        x = pos_logit - log_denom
        log_eps = x.new_full((), math.log(1e-5))
        return (-torch.logaddexp(x, log_eps)).mean()

    # -------- Importance-sampled path: O(B*K) --------
    K = int(neg_k)

    # Build proposal q over batch indices (data-driven)
    if sample_weight is None:
        q = torch.full((B,), 1.0 / B, device=u.device, dtype=u.dtype)
    else:
        w = sample_weight.to(device=u.device, dtype=u.dtype).clamp_min(0.0)
        w_sum = w.sum()
        if float(w_sum) <= 0.0:
            q_data = torch.full((B,), 1.0 / B, device=u.device, dtype=u.dtype)
        else:
            q_data = w / (w_sum + eps)

        q_uni = torch.full((B,), 1.0 / B, device=u.device, dtype=u.dtype)
        lam = float(mix_uniform)
        lam = 0.0 if lam < 0.0 else (1.0 if lam > 1.0 else lam)
        q = lam * q_uni + (1.0 - lam) * q_data

        # avoid tiny probabilities
        q = q.clamp_min(1e-8)
        q = q / q.sum()

    # Sample K indices (with replacement), shared across all anchors in the batch
    q_cpu = q.detach().float().cpu()          # 采样不需要梯度，detach 更安全
    idx = torch.multinomial(q_cpu, K, replacement=True)  # CPU 上确定性
    idx = idx.to(q.device, non_blocking=True)
    #idx = torch.multinomial(q, K, replacement=True)  # [K]
    cand_neg = cand.index_select(0, idx)             # [K, d]
    log_q = torch.log(q.index_select(0, idx) + eps)  # [K]

    # scores: [B, K]
    scores = torch.matmul(u, cand_neg.t()) / tau
    # importance correction: exp(score)/q  <=> score - log q
    log_terms = scores - log_q.view(1, -1)

    # log Z_hat = log( (1/K) * sum exp(log_terms) )
    logZ_hat = torch.logsumexp(log_terms, dim=1) - math.log(K)

    x = pos_logit - logZ_hat
    log_eps = x.new_full((), math.log(1e-5))
    return (-torch.logaddexp(x, log_eps)).mean()




def get_sampled_neighbor_aggregate_loss(
    embedding1, embedding2, tau,
    num_neg=64, cand_mult=4, hard_frac=0.5,
    pop_log=None, pop_tau=0.7,
    # ===== 新增（若提供则走新方法）=====
    all_item_emb=None,
    item_sampling_prob=None,
    pos_ids=None,
    fn_delta=0.95,
    user_neg=32
):
    """
    三条路径：
    A) 旧版（in-batch per-row candidates + topk），保持兼容
    B) 全局 item 采样（importance correction + FN filter + user-user negatives）
    C) 【推荐用于你当前目标】分层列采样（接近 keep_ratio * B * B）：
       - 分母保持 “exp(u·(v+u))” 的原结构
       - 对角项 exp(u_i·(v_i+u_i)) 精确补回，避免采样漏掉关键项
       - hard columns（anchor_k 个）+ random columns（补齐到 keep_ratio*B）
       - 随机部分做逐行无偏缩放，降低方差，通常能把差的 5% 找回来
    """

    # ========== C) 分层列采样：接近 keep_ratio * B * B ==========
    if keep_ratio is not None:
        device = embedding1.device
        u = F.normalize(embedding1, dim=-1)
        v = F.normalize(embedding2, dim=-1)

        B = u.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        kr = float(keep_ratio)
        if kr >= 1.0:
            return get_neighbor_aggregate_loss(embedding1, embedding2, tau)
        if kr <= 0.0:
            kr = 0.8

        m = int(round(kr * B))
        m = max(2, min(B, m))

        cand = v + u  # [B, d]

        # Shared column subset J
        if (col_idx is None) or (col_idx.numel() != m) or (col_idx.device != device):
            if block_sampling:
                start = torch.randint(0, B, (1,), device=device, dtype=torch.long)  # tensor, no CPU sync
                col_idx = (start + torch.arange(m, device=device, dtype=torch.long)) % B
            else:
                col_idx = torch.randperm(B, device=device)[:m]

        cand_sub = cand[col_idx]  # [m, d]
        logits = torch.matmul(u, cand_sub.t()) / tau  # [B, m]

        # Exclude diagonal column (j=i) from the sampled off-diagonal estimate
        pos_map = torch.full((B,), -1, device=device, dtype=torch.long)
        pos_map[col_idx] = torch.arange(m, device=device, dtype=torch.long)
        row = torch.arange(B, device=device, dtype=torch.long)
        col = pos_map[row]
        in_subset = col >= 0
        if in_subset.any():
            logits[row[in_subset], col[in_subset]] = float('-inf')

        # log off estimate: log( (B/m) * sum_{j in J, j≠i} exp(logit_ij) )
        lse_off = torch.logsumexp(logits, dim=1)  # [B]
        log_scale = lse_off.new_full((), math.log(B / float(m)))
        log_off_est = lse_off + log_scale

        # exact diagonal: log exp( u_i·(v_i+u_i) / tau )
        diag_logit = torch.sum(u * (v + u), dim=1) / tau  # [B]

        # log denom = logaddexp(diag, off_est)
        log_denom = torch.logaddexp(diag_logit, log_off_est)

        # numerator: exp(u_i·v_i / tau)
        pos_logit = torch.sum(u * v, dim=1) / tau
        x = pos_logit - log_denom
        log_eps = x.new_full((), math.log(1e-5))
        loss = -torch.logaddexp(x, log_eps)
        return loss.mean()

    # ========== B) 你文件里已有的“全局 item 采样”路径 ==========
    if (all_item_emb is not None) and (item_sampling_prob is not None) and (pos_ids is not None):
        device = embedding1.device
        eps = 1e-12

        u = F.normalize(embedding1, dim=-1)          # [B, d]
        v_pos = F.normalize(embedding2, dim=-1)      # [B, d]

        B = u.shape[0]
        if B <= 1:
            return torch.tensor(0.0, device=device)

        k = int(min(num_neg, B - 1))
        m = int(max(1, cand_mult * k))

        n_items = all_item_emb.shape[0]
        m = int(min(m, n_items))
        if m <= 0:
            return torch.tensor(0.0, device=device)

        replacement = False
        if n_items <= 1:
            return torch.tensor(0.0, device=device)

        cand_ids = torch.multinomial(item_sampling_prob, num_samples=m, replacement=replacement)  # [m]
        cand_emb = F.normalize(all_item_emb[cand_ids], dim=-1)                                    # [m, d]

        logits_item = torch.matmul(u, cand_emb.t()) / tau
        cos_item = logits_item * tau

        if fn_delta is not None:
            logits_item = logits_item.masked_fill(cos_item > float(fn_delta), float('-inf'))

        same_as_pos = (pos_ids.view(-1, 1) == cand_ids.view(1, -1))
        logits_item = logits_item.masked_fill(same_as_pos, float('-inf'))

        log_q = torch.log(item_sampling_prob[cand_ids] + eps)  # [m]
        logits_item = logits_item - log_q.view(1, -1)

        Ku = int(min(max(0, user_neg), B - 1))
        if Ku > 0:
            idx = torch.randint(0, B - 1, (B, Ku), device=device)
            arange = torch.arange(B, device=device).unsqueeze(1)
            idx = idx + (idx >= arange).long()
            u_cand = u[idx]  # [B, Ku, d]
            logits_user = torch.bmm(u_cand, u.unsqueeze(2)).squeeze(2) / tau  # [B, Ku]
        else:
            logits_user = None

        pos_logit = torch.sum(u * v_pos, dim=1, keepdim=True) / tau  # [B, 1]

        if logits_user is None:
            logits = torch.cat([pos_logit, logits_item], dim=1)
        else:
            logits = torch.cat([pos_logit, logits_item, logits_user], dim=1)

        loss = (-pos_logit.squeeze(1) + torch.logsumexp(logits, dim=1)).mean()
        return loss

    # ========== A) 旧版兼容路径（你的原实现） ==========
    embedding1 = torch.nn.functional.normalize(embedding1)
    embedding2 = torch.nn.functional.normalize(embedding2)

    batch_size = embedding1.shape[0]
    if batch_size <= 1:
        return torch.tensor(0.0, device=embedding1.device)

    pos_logits = torch.sum(embedding1 * embedding2, dim=1, keepdim=True) / tau

    k = min(num_neg, batch_size - 1)
    m = min(cand_mult * k, batch_size - 1)

    cand_idx = torch.randint(0, batch_size - 1, (batch_size, m), device=embedding1.device)
    arange = torch.arange(batch_size, device=embedding1.device).unsqueeze(1)
    cand_idx = cand_idx + (cand_idx >= arange).long()

    cand_embed = embedding2[cand_idx]  # [B, M, d]
    cand_logits = torch.bmm(cand_embed, embedding1.unsqueeze(2)).squeeze(2) / tau

    hard_k = max(1, int(k * hard_frac)) if k > 1 else 1
    hard_k = min(hard_k, k)
    hard_vals, hard_idx = torch.topk(cand_logits, k=hard_k, dim=1)

    rep_k = k - hard_k
    if rep_k > 0:
        if pop_log is None:
            weights = torch.ones_like(cand_logits)
        else:
            cand_pop = pop_log[cand_idx]  # [B, M]
            pop_center = cand_pop.median(dim=1, keepdim=True).values
            weights = torch.exp(-torch.abs(cand_pop - pop_center) / max(pop_tau, 1e-6))

        weights.scatter_(1, hard_idx, 0.0)
        weight_sum = weights.sum(dim=1, keepdim=True)
        weights = torch.where(
            weight_sum > 0,
            weights / weight_sum,
            torch.full_like(weights, 1.0 / weights.shape[1])
        )
        rep_idx = torch.multinomial(weights, rep_k, replacement=True)
        rep_vals = cand_logits.gather(1, rep_idx)
        neg_logits = torch.cat([hard_vals, rep_vals], dim=1)
    else:
        neg_logits = hard_vals

    logits = torch.cat([pos_logits, neg_logits], dim=1)
    labels = torch.zeros(batch_size, dtype=torch.long, device=embedding1.device)
    return torch.nn.functional.cross_entropy(logits, labels)


def get_noise_filtered_infonce(user_emb, pos_emb, all_item_emb, tau=0.2, rho=0.3, delta=0.9):
    user_emb = F.normalize(user_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)
    all_item_emb = F.normalize(all_item_emb, dim=-1)

    pos_sim = torch.sum(user_emb * pos_emb, dim=1) / tau
    all_sim = torch.matmul(user_emb, all_item_emb.T) / tau  # [B, N_items]

    hard_mask = (all_sim < delta)

    k = max(1, int(rho * all_sim.shape[1]))
    topk_vals, _ = torch.topk(all_sim, k=k, largest=True)
    thresh = topk_vals[:, -1].unsqueeze(1)
    hard_mask &= (all_sim >= thresh)

    exp_pos = torch.exp(pos_sim)
    exp_negs = torch.exp(all_sim) * hard_mask.float()
    denom = exp_negs.sum(dim=1) + exp_pos
    loss = -torch.log(exp_pos / denom)
    return loss.mean()
