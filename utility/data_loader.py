import random

import scipy.sparse as sp
import numpy as np

class Data(object):
    def __init__(self, args):
        self.args = args
        self.path = self.args.dataset_path + args.dataset
        self.filetype = self.args.dataset_type
        self.num_users = 0
        self.num_items = 0
        self.num_nodes = 0
        self.load_data_and_create_sp()

        # ===== 新增：构建 item–item 共现 KNN，用于 diffused NA =====
        # K 由超参数 item_knn_k 控制，默认为 10
        #self.build_item_knn()

        if int(args.sparsity_test) == 1:
            self.split_test_dict, self.split_state = self.create_sparsity_split()

    def load_data_and_create_sp(self):
        train_path = self.path + "/train" + self.filetype
        test_path = self.path + "/test" + self.filetype

        self.unique_train_users, self.train_users, self.train_items, self.train_pos_len, self.train_num_inter, self.train_dict = self.read_file(train_path)
        self.unique_test_users,  self.test_users,  self.test_items,  self.test_pos_len,  self.test_num_inter, self.test_dict = self.read_file(test_path)
        assert len(self.train_users) == len(self.train_items)

        self.num_users += 1
        self.num_items += 1
        self.num_nodes = self.num_users + self.num_items

        # U*I
        self.train_mat = sp.coo_matrix(
            (np.ones(len(self.train_users)), (self.train_users, self.train_items)),
            shape=[self.num_users, self.num_items]
        )
        self.test_mat  = sp.coo_matrix(
            (np.ones(len(self.test_users)),  (self.test_users, self.test_items)),
            shape=[self.num_users, self.num_items]
        )

        self.all_positive = self.get_user_pos_items(list(range(self.num_users)))

    # removed popularity sampler; back to uniform sampling

    def read_file(self, file_name):
        inter_users, inter_items, unique_user, user_dict = [], [], [], {}
        pos_length = []
        num_inter = 0
        with open(file_name, "r") as f:
            line = f.readline()
            while line is not None and line != "":
                temp = line.strip()
                arr = [int(i) for i in temp.split(" ")]
                user_id, pos_id = arr[0], arr[1:]

                self.num_users = max(self.num_users, user_id)
                self.num_items = max(self.num_items, max(pos_id))

                unique_user.append(user_id)

                inter_users.extend([user_id] * len(pos_id))
                inter_items.extend(pos_id)

                pos_length.append(len(pos_id)) # [10, 20, 10, 15]
                num_inter += len(pos_id)

                for i in range(0, len(pos_id)):
                    if i == 0:
                        user_dict[user_id] = [pos_id[i]]
                    else:
                        user_dict[user_id].append(pos_id[i])

                line = f.readline()

        return np.array(unique_user), np.array(inter_users), np.array(inter_items), pos_length, num_inter, user_dict

    def random_create_user_pos_neg(self):
        pairs = []

        for i in range(len(self.train_users)):
            user = self.train_users[i]
            pos_items = self.train_dict[user]
            if len(pos_items) == 0:
                continue

            pos_item = self.train_items[i]
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in pos_items:
                    break
            pairs.append([user, pos_item, neg_item])
        return np.array(pairs)

    def random_create_user_pos_neg_cl(self):
        pairs = []
        for i in range(len(self.train_users)):
            user = self.train_users[i]
            pos_items = self.train_dict[user]
            if len(pos_items) == 0:
                continue

            pos_item = self.train_items[i]
            while True:
                pos_item2 = random.sample(pos_items, k=1)[0]
                if pos_item2 != pos_item:
                    break
            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in pos_items:
                    break
            pairs.append([user, pos_item, neg_item, pos_item2])
        return np.array(pairs)

    def sparse_adjacency_matrix(self):
        """
        构造归一化邻接矩阵：
        - use_pop_adj = 0: 原版 LightCCF 邻域（所有边权相同）
        - use_pop_adj = 1: Popularity-Calibrated 邻域（按 item 度数做列缩放）
        """
        # 是否使用 popularity 校准
        use_pop_adj = int(getattr(self.args, 'use_pop_adj', 0))
        pop_alpha   = float(getattr(self.args, 'pop_alpha', 0.3))

        if use_pop_adj:
            # 不同 alpha 保存到不同文件，避免互相覆盖
            adj_name = f'/pre_Adj_pop_alpha{pop_alpha:.2f}.npz'
        else:
            adj_name = '/pre_Adj.npz'

        adj_path = self.path + adj_name

        try:
            normal_adjacency = sp.load_npz(adj_path)
            print(f'\t Adjacency matrix exist. Now loading from {adj_path}!')
        except:
            print(f'\t Adjacency matrix not exist. Now constructing ({adj_name})!')

            # === 1) 基础大邻接 A ===
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()

            # 原始用户-物品交互矩阵 R (U x I)
            R = self.train_mat.tocsr()

            if use_pop_adj:
                # ---------- Popularity-Calibrated 列缩放 ----------
                # item 度：每个物品被交互的次数
                item_deg = np.array(R.sum(axis=0)).flatten().astype(np.float32)  # [I]
                eps = 1e-6
                w = np.power(item_deg + eps, -pop_alpha)                          # [I]
                W = sp.diags(w)                                                  # I x I
                R_weighted = R.dot(W)                                            # U x I

                print(f'\t Using popularity-calibrated adjacency (alpha={pop_alpha})')
            else:
                # 原始 R，所有边权相同
                R_weighted = R
                print('\t Using original unweighted adjacency.')

            # U-I, I-U 两块
            adjacency_matrix[:self.num_users, self.num_users:] = R_weighted
            adjacency_matrix[self.num_users:, :self.num_users] = R_weighted.T

            adjacency_matrix = adjacency_matrix.tocsr()

            # === 2) D^{-1/2} A D^{-1/2} 归一化 ===
            row_sum = np.array(adjacency_matrix.sum(axis=1)).flatten()
            d_inv = np.power(row_sum + 1e-7, -0.5)
            d_inv[np.isinf(d_inv)] = 0.
            d_inv[np.isnan(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(adj_path, normal_adjacency)
            print(f'\t Adjacency matrix constructed and saved to {adj_path}.')

        return normal_adjacency

    def sparse_adjacency_matrix_self(self):
        try:
            normal_adjacency = sp.load_npz(self.path + '/pre_Adj_self.npz')
            print('\t Adjacency matrix exist. Now loading!')
        except:
            print('\t Adjacency matrix not exist. Now constructing!')
            adjacency_matrix = sp.dok_matrix((self.num_nodes, self.num_nodes), dtype=np.float32)
            adjacency_matrix = adjacency_matrix.tolil()
            R = self.train_mat.todok()
            # adjacency_matrix[row1:row2, column1:column2]
            adjacency_matrix[:self.num_users, self.num_users:] = R
            adjacency_matrix[self.num_users:, :self.num_users] = R.T

            # add self
            adjacency_matrix = adjacency_matrix.todok()
            adjacency_matrix = adjacency_matrix + sp.eye(adjacency_matrix.shape[0])

            # A_hat = D^(-1/2) A D(-1/2)
            row_sum = np.array(adjacency_matrix.sum(axis=1))
            d_inv = np.power(row_sum, -0.5).flatten()
            d_inv[np.isinf(d_inv)] = 0.
            degree_matrix = sp.diags(d_inv)

            normal_adjacency = degree_matrix.dot(adjacency_matrix).dot(degree_matrix).tocsr()
            sp.save_npz(self.path + '/pre_Adj_self', normal_adjacency)
            print('\t Adjacency matrix constructed.')
        return normal_adjacency

    def user_item_num(self):
        return self.num_users, self.num_items

    def create_sparsity_split(self):
        all_users = list(self.test_dict.keys())
        user_n_iid = dict()

        for uid in all_users:
            train_iids = self.all_positive[uid]
            test_iids = self.test_dict[uid]

            num_iids = len(train_iids) + len(test_iids)

            if num_iids not in user_n_iid.keys():
                user_n_iid[num_iids] = [uid]
            else:
                user_n_iid[num_iids].append(uid)

        split_uids = list()
        temp = []
        count = 1
        fold = 3
        # fold = 4
        n_count = self.train_num_inter + self.test_num_inter
        n_rates = 0
        split_state = []
        for idx, n_iids in enumerate(sorted(user_n_iid)):
            temp += user_n_iid[n_iids]
            n_rates += n_iids * len(user_n_iid[n_iids])
            n_count -= n_iids * len(user_n_iid[n_iids])

            if n_rates >= count * 0.334 * (self.train_num_inter + self.test_num_inter):
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

                temp = []
                n_rates = 0
                fold -= 1

            if idx == len(user_n_iid.keys()) - 1 or n_count == 0:
                split_uids.append(temp)
                state = '\t #inter per user<=[%d], #users=[%d], #all rates=[%d]' % (n_iids, len(temp), n_rates)
                split_state.append(state)
                print(state)

        return split_uids, split_state

    def get_user_pos_items(self, users):
        self.train_mat_csr = self.train_mat.tocsr()
        positive_items = []
        for user in users:
            positive_items.append(self.train_mat_csr[user].nonzero()[1])
        return positive_items
    """
    # ===== 新增：item–item 共现 KNN，用于 diffused NA =====
        # ===== 新增：item–item 共现 KNN，用于 diffused NA（优化版） =====
    def build_item_knn(self):
        K = int(getattr(self.args, 'item_knn_k', 10))  # 默认为 10 个邻居
        if K <= 0:
            print('\t item_knn_k <= 0, skip building item KNN.')
            self.item_knn = None
            return

        save_path = self.path + f'/item_knn_k{K}_dict.npy'
        try:
            self.item_knn = np.load(save_path)
            print(f'\t item_knn exist (K={K}). Now loading!')
            return
        except Exception:
            print(f'\t item_knn not exist. Now constructing with K={K}!')

        # ------ 参数：控制极大度用户的截断，防止单个用户交互太多导致 O(deg^2) 爆炸 ------
        max_deg = int(getattr(self.args, 'item_knn_max_degree', 200))
        # 例如：用户交互数 > 200 时，只随机取 200 个 item 来做共现统计

        # 为每个 item 准备一个 dict：neighbor_id -> count
        co_dict = [dict() for _ in range(self.num_items)]

        # self.train_dict: user -> [item1, item2, ...]
        # 这里直接用 train_dict，避免额外转换
        for u, items in self.train_dict.items():
            if len(items) <= 1:
                continue

            arr = np.array(items, dtype=np.int64)
            L = len(arr)
            # 截断大度数用户
            if L > max_deg:
                idx = np.random.choice(L, size=max_deg, replace=False)
                arr = arr[idx]
                L = max_deg

            # 累计共现：对 (i, j) 成对计数
            for x in range(L):
                i = int(arr[x])
                di = co_dict[i]
                for y in range(L):
                    if x == y:
                        continue
                    j = int(arr[y])
                    di[j] = di.get(j, 0) + 1

        # ------ 为每个 item 取共现 top-K 邻居 ------
        item_knn = np.zeros((self.num_items, K), dtype=np.int64)
        for i in range(self.num_items):
            neigh = co_dict[i]
            if len(neigh) == 0:
                # 完全没有共现邻居时，随机填充
                item_knn[i] = np.random.randint(0, self.num_items, size=K)
                continue

            ids = np.fromiter(neigh.keys(), dtype=np.int64)
            vals = np.fromiter(neigh.values(), dtype=np.int64)

            if len(ids) > K:
                # 取共现次数最大的前 K 个
                top_idx = np.argpartition(-vals, K - 1)[:K]
                ids = ids[top_idx]
            # 不足 K 个邻居就随便 pad 一些
            if len(ids) < K:
                pad = np.random.randint(0, self.num_items, size=K - len(ids))
                ids = np.concatenate([ids, pad])

            item_knn[i] = ids

        np.save(save_path, item_knn)
        self.item_knn = item_knn
        print('\t item_knn constructed (dict-based).')
"""
