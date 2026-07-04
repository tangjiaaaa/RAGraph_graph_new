from utils.parse_args import args
import numpy as np
import scipy.sparse as sp
import torch
import logging
from copy import deepcopy
from collections import defaultdict
import pandas as pd
import networkx as nx
from .complex import Cochain, Complex, ComplexBatch
logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger('train_logger')
logger.setLevel(logging.INFO)


class GraphStructure:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.cochains = {}

    def _extract_ring_structure(self):
        """
        提取全局图的环结构，生成 ComplexBatch
        """
        G = nx.Graph()
        G.add_edges_from(self.edgelist.tolist())
        cycles = list(nx.cycle_basis(G))
        ring_edges = []
        for cycle in cycles:
            for i in range(len(cycle)):
                u = cycle[i]
                v = cycle[(i + 1) % len(cycle)]
                ring_edges.append([u, v])
        ring_boundary_index = torch.tensor(ring_edges, dtype=torch.long).T if ring_edges else torch.empty((2, 0),
                                                                                                          dtype=torch.long)
        cochain2 = Cochain(dim=2, boundary_index=ring_boundary_index)

        node_features = torch.arange(G.number_of_nodes(), dtype=torch.float32).unsqueeze(1)
        edge_index = torch.tensor(list(G.edges), dtype=torch.long).T if G.number_of_edges() > 0 else torch.empty((2, 0),
                                                                                                                 dtype=torch.long)
        cochain0 = Cochain(dim=0, x=node_features)
        cochain1 = Cochain(dim=1, upper_index=edge_index)
        complex_obj = Complex(cochain0, cochain1, cochain2, dimension=2)
        self.graphs = ComplexBatch.from_complex_list([complex_obj])

        logger.info(f"Extracted {len(cycles)} cycles in the global graph.")

    @staticmethod
    def generate_complex_batch(graphs):
        complex_batch = []
        for G in graphs:
            gs = GraphStructure(G)
            ring_cochain = gs.extract_ring_structure()

            # 节点特征：如果没有特征，用索引占位
            node_features = torch.arange(G.number_of_nodes(), dtype=torch.float32).unsqueeze(1)
            edge_index = torch.tensor(list(G.edges), dtype=torch.long).T if G.number_of_edges() > 0 else torch.empty((2,0), dtype=torch.long)

            cochain0 = Cochain(dim=0, x=node_features)
            cochain1 = Cochain(dim=1, upper_index=edge_index)
            complex_obj = Complex(cochain0, cochain1, ring_cochain, dimension=2)
            complex_batch.append(complex_obj)
        return ComplexBatch.from_complex_list(complex_batch)



class EdgeListData:
    def __init__(self, train_file, test_file, phase='pretrain', pre_dataset=None, user_hist_files=[], has_time=True):
        logger.info(f"Loading dataset for {phase}...")
        self.phase = phase
        self.has_time = has_time
        self.pre_dataset = pre_dataset

        self.hour_interval = args.hour_interval_pre if phase == 'pretrain' else args.hour_interval_f

        self.edgelist = []
        self.edge_time = []
        self.num_users = 0
        self.num_items = 0
        self.num_edges = 0

        self.train_user_dict = {}
        self.test_user_dict = {}

        self._load_data(train_file, test_file, has_time)
        self._extract_ring_structure()  # 提取每个图的环结构
        if phase == 'pretrain':
            self.user_hist_dict = self.train_user_dict
        elif phase == 'finetune':
            self.user_hist_dict = deepcopy(self.train_user_dict)
            self._load_user_hist_from_files(user_hist_files)

        users_has_hist = set(list(self.user_hist_dict.keys()))
        all_users = set(list(range(self.num_users)))
        users_no_hist = all_users - users_has_hist
        logger.info(f"Number of users from all users with no history: {len(users_no_hist)}")
        for u in users_no_hist:
            self.user_hist_dict[u] = []

    def fast_cycle_basis(self, G: nx.Graph):
        """
        Fast Cycle Basis: 基于生成树 + 非树边构建最小环基
        复杂度 O(N+M)
        """
        parent = {n: None for n in G.nodes()}
        visited = set()
        order = []

        # 构建 DFS 树
        for start in G.nodes():
            if start in visited:
                continue
            stack = [(start, None)]
            while stack:
                node, par = stack.pop()
                if node in visited:
                    continue
                visited.add(node)
                parent[node] = par
                order.append(node)
                for nbr in G[node]:
                    if nbr != par:
                        stack.append((nbr, node))

        # 找非树边：形成基环
        tree_edges = set()
        for node in G.nodes():
            if parent[node] is not None:
                tree_edges.add((min(node, parent[node]), max(node, parent[node])))

        cycles = []
        for u, v in G.edges():
            edge = (min(u, v), max(u, v))
            if edge in tree_edges:
                continue
            # 追溯 u->root, v->root，找到路径
            path_u = []
            path_v = []
            uu, vv = u, v
            while uu is not None:
                path_u.append(uu)
                uu = parent[uu]
            while vv is not None:
                path_v.append(vv)
                vv = parent[vv]
            # 找最低公共祖先
            i = len(path_u) - 1
            j = len(path_v) - 1
            while i >= 0 and j >= 0 and path_u[i] == path_v[j]:
                i -= 1
                j -= 1
            # 组成环
            cycle = path_u[:i + 2] + path_v[:j + 1][::-1]
            cycles.append(cycle)
        return cycles

    @staticmethod
    def split_graph_nodes(total_nodes, chunk_size):
        """按节点范围拆分，返回 [start, end) 区间"""
        chunks = []
        for i in range(0, total_nodes, chunk_size):
            chunks.append((i, min(i + chunk_size, total_nodes)))
        return chunks

    def build_complex_batch(self, chunk_size=1000, max_rings_per_chunk=200):
        """
        分批提取子图的 Fast Cycle Basis，避免大图 OOM
        返回: complex_batch, batch_vector, all_global_node_idx
        """
        G = nx.Graph()
        G.add_edges_from(self.edgelist.tolist())
        node_chunks = self.split_graph_nodes(G.number_of_nodes(), chunk_size)
        complex_list = []
        batch_vector = []
        all_global_node_idx = []  # ← 初始化

        for idx, (start, end) in enumerate(node_chunks):
            sub_nodes = [n for n in G.nodes() if start <= n < end and G.degree[n] > 0]
            if not sub_nodes:  # 跳过空子图
                continue
            G_sub = G.subgraph(sub_nodes).copy()
            global_node_idx = torch.tensor(sub_nodes, dtype=torch.long)
            all_global_node_idx.append(global_node_idx)  # 收集每个子图的全局节点索引

            # 重新编号子图节点
            mapping = {n: i for i, n in enumerate(G_sub.nodes())}
            G_sub = nx.relabel_nodes(G_sub, mapping)

            # print(f"[DEBUG] Subgraph {idx}: nodes={G_sub.number_of_nodes()}, edges={G_sub.number_of_edges()}")

            # 提取环
            cycles = self.fast_cycle_basis(G_sub)
            if len(cycles) > max_rings_per_chunk:
                # print(f"[DEBUG] Subgraph {idx}: rings {len(cycles)} > {max_rings_per_chunk}, truncating")
                cycles = cycles[:max_rings_per_chunk]

            # 构建环边界
            ring_edges = []
            for cycle in cycles:
                for i in range(len(cycle)):
                    ring_edges.append([cycle[i], cycle[(i + 1) % len(cycle)]])
            ring_boundary_index = torch.tensor(ring_edges, dtype=torch.long).T if ring_edges else torch.empty((2, 0),
                                                                                                              dtype=torch.long)



            # 节点 cochain
            node_features = torch.arange(G_sub.number_of_nodes(), dtype=torch.float32).unsqueeze(1)
            cochain0 = Cochain(dim=0)
            cochain0.num_cells = G_sub.number_of_nodes()
            cochain0.x = node_features
            cochain0.upper_index = None
            cochain0.boundary_index = None

            # 边 cochain
            edge_index = torch.tensor(list(G_sub.edges),
                                      dtype=torch.long).T if G_sub.number_of_edges() > 0 else torch.empty((2, 0),
                                                                                                          dtype=torch.long)
            cochain1 = Cochain(dim=1)
            cochain1.upper_index = None  # ← 这里设置为 None
            cochain1.boundary_index = edge_index  # 边界 = 两个节点            cochain1.boundary_index = None
            cochain2 = Cochain(dim=2)
            cochain2.boundary_index = ring_boundary_index
            cochain2.upper_index = None
            # 组合 Complex
            complex_obj = Complex(cochain0, cochain1, cochain2, dimension=2)
            complex_list.append(complex_obj)
            batch_vector += [idx] * G_sub.number_of_nodes()

        if not complex_list:  # 防止为空
            raise ValueError("No valid subgraphs found for ComplexBatch.")

        complex_batch = ComplexBatch.from_complex_list(complex_list)
        batch_vector = torch.tensor(batch_vector, dtype=torch.long)
        all_global_node_idx = torch.cat(all_global_node_idx, dim=0)  # 拼接所有子图节点索引

        # print(f"[DEBUG] ComplexBatch built: {len(complex_list)} subgraphs, total nodes={len(batch_vector)}")
        return complex_batch, batch_vector, all_global_node_idx

    def _extract_ring_structure(self):
        pass
        # """
        # 提取每个图的环结构，保存至 self.graphs
        # """
        # # 确保你已经加载了图的数据并且生成了图结构
        # self.graphs = []  # 存储所有的图
        # for i in range(len(self.edgelist)):
        #     graph = self.create_graph(i)  # 根据图的索引生成图数据
        #     graph.extract_ring_structure()  # 提取图中的环结构
        #     self.graphs.append(graph)  # 将提取的图添加到 graph 集合中

    def create_graph(self, graph_idx):
        """
        根据边列表构建 NetworkX 图（即便包含 0 节点/越界节点，也自动处理）
        """
        G = nx.Graph()
        for edge in self.edgelist:
            if isinstance(edge, (list, tuple, np.ndarray)) and len(edge) == 2:
                u, v = int(edge[0]), int(edge[1])
                # 越界节点或特殊节点，映射到虚拟节点（避免报错）
                if u < 0 or v < 0:
                    continue
                if u >= self.num_users:
                    u = self.num_users + (u - self.num_users)  # 重新映射
                if v >= self.num_items:
                    v = self.num_items + (v - self.num_items)
                G.add_edge(u, v)
            else:
                continue
        return G

    def _read_file(self, train_file, test_file, has_time=True):
        with open(train_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                if not has_time:
                    user, items = line[:2]
                    times = " ".join(["0"] * len(items.split(" ")))
                else:
                    user, items, times = line

                for item in items.split(" "):
                    self.edgelist.append((int(user), int(item)))
                for time in times.split(" "):
                    self.edge_time.append(int(time))
                self.train_user_dict[int(user)] = [int(item) for item in items.split(" ")]

        self.test_edge_num = 0
        with open(test_file, 'r') as f:
            for line in f:
                line = line.strip().split('\t')
                user, items = line[:2]
                self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
                self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))

    def _read_pd(self, train_pd, test_pd, has_time=True):
        for i in range(len(train_pd)):
            line = train_pd.iloc[i]
            if not has_time:
                user, items = line.iloc[0], line.iloc[1]
                times = " ".join(["0"] * len(items.split(" ")))
            else:
                user, items, times = line.iloc[0], line.iloc[1], line.iloc[2]

            for i in items.split(" "):
                self.edgelist.append((int(user), int(i)))
            for i in times.split(" "):
                self.edge_time.append(int(i))
            self.train_user_dict[int(user)] = [int(i) for i in items.split(" ")]

        self.test_edge_num = 0
        for i in range(len(test_pd)):
            line = test_pd.iloc[i]
            user, items = line.iloc[0], line.iloc[1]
            self.test_user_dict[int(user)] = [int(i) for i in items.split(" ")]
            self.test_edge_num += len(self.test_user_dict[int(user)])
        logger.info('Number of test users: {}'.format(len(self.test_user_dict)))

    def _load_data(self, train_file, test_file, has_time=True):
        if isinstance(train_file, pd.DataFrame):
            self._read_pd(train_file, test_file, has_time)
        else:
            self._read_file(train_file, test_file, has_time)

        # === 1. 转 numpy ===
        self.edgelist = np.array(self.edgelist, dtype=np.int32)
        self.edge_time = 1 + self.timestamp_to_time_step(np.array(self.edge_time, dtype=np.int32))
        self.num_edges = len(self.edgelist)

        # === 2. 计算 num_users & num_items：用全量索引防止越界 ===
        if self.pre_dataset is not None:
            self.num_users = self.pre_dataset.num_users
            self.num_items = self.pre_dataset.num_items
        else:
            all_user_ids = list(self.edgelist[:, 0]) + list(self.test_user_dict.keys())
            all_item_ids = list(self.edgelist[:, 1]) + [i for items in self.test_user_dict.values() for i in items]
            self.num_users = max(all_user_ids) + 1
            self.num_items = max(all_item_ids) + 1

        # === 3. 不做 item 偏移，保持 user × item 矩阵 ===
        total_nodes = self.num_users + self.num_items

        # === 4. 打印统计 ===
        logger.info(f'Number of users: {self.num_users}')
        logger.info(f'Number of items: {self.num_items}')
        logger.info(f'Number of edges: {self.num_edges}')
        logger.info(f'Total graph nodes: {total_nodes}')

        # === 5. 构建用户-物品矩阵 ===
        # 注意：行 = 用户索引，列 = 物品索引
        self.graph = sp.coo_matrix(
            (np.ones(self.num_edges), (self.edgelist[:, 0], self.edgelist[:, 1])),
            shape=(self.num_users, self.num_items)
        )

        # === 6. 越界检查 ===
        if self.edgelist[:, 0].max() >= self.num_users or self.edgelist[:, 1].max() >= self.num_items:
            raise ValueError(f"[ERROR] EdgeList contains out-of-range ids: "
                             f"user_max={self.edgelist[:, 0].max()}, num_users={self.num_users}, "
                             f"item_max={self.edgelist[:, 1].max()}, num_items={self.num_items}")

        # print(f"[DEBUG] graph shape: {self.graph.shape}, num_users={self.num_users}, num_items={self.num_items}")

        # === 7. 时间字典 ===
        if self.has_time:
            self.edge_time_dict = defaultdict(dict)
            for i in range(len(self.edgelist)):
                u, v = self.edgelist[i]
                v_offset = v + self.num_users  # 物品节点偏移
                self.edge_time_dict[u][v_offset] = self.edge_time[i]
                self.edge_time_dict[v_offset][u] = self.edge_time[i]
                # print(f"[DEBUG] edge_time_dict sample: {list(self.edge_time_dict.items())[:3]}")


    def _load_user_hist_from_files(self, user_hist_files):
        for file in user_hist_files:
            with open(file, 'r') as f:
                for line in f:
                    line = line.strip().split('\t')
                    user, items = int(line[0]), [int(i) for i in line[1].split(" ")]
                    try:
                        self.user_hist_dict[user].extend(items)
                    except KeyError:
                        self.user_hist_dict[user] = items

    def sample_subgraph(self):
        pass

    def get_train_batch(self, start, end):

        def negative_sampling(user_item, train_user_set, n=1):
            neg_items = []
            for user, _ in user_item:
                user = int(user)
                for _ in range(n):
                    while True:
                        neg_item = np.random.randint(low=0, high=self.num_items, size=1)[0]
                        if neg_item not in train_user_set[user]:
                            break
                    neg_items.append(neg_item)
            return neg_items

        ui_pairs = self.edgelist[start:end]
        users = torch.LongTensor(ui_pairs[:, 0]).to(args.device)
        pos_items = torch.LongTensor(ui_pairs[:, 1]).to(args.device)
        if args.model == "MixGCF":
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, args.n_negs)
        else:
            neg_items = negative_sampling(ui_pairs, self.train_user_dict, 1)
        neg_items = torch.LongTensor(neg_items).to(args.device)
        return users, pos_items, neg_items

    def shuffle(self):
        random_idx = np.random.permutation(self.num_edges)
        self.edgelist = self.edgelist[random_idx]
        self.edge_time = self.edge_time[random_idx]

    def _generate_binorm_adj(self, edgelist):
        adj = sp.coo_matrix((np.ones(len(edgelist)), (edgelist[:, 0], edgelist[:, 1])),
                            shape=(self.num_users, self.num_items), dtype=np.float32)

        a = sp.csr_matrix((self.num_users, self.num_users))
        b = sp.csr_matrix((self.num_items, self.num_items))
        adj = sp.vstack([sp.hstack([a, adj]), sp.hstack([adj.transpose(), b])])
        adj = (adj != 0) * 1.0
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        adj = adj.dot(d_inv_sqrt_mat).transpose().dot(d_inv_sqrt_mat).tocoo()

        ui_adj = adj.tocsr()[:self.num_users, self.num_users:].tocoo()
        return adj

    def timestamp_to_time_step(self, timestamp_arr, least_time=None):
        interval_hour = self.hour_interval
        if least_time is None:
            least_time = np.min(timestamp_arr)
            print("1st least time: ", least_time)
            print("2nd least time: ", np.sort(timestamp_arr)[1])
            print("3rd least time: ", np.sort(timestamp_arr)[2])
            print("Max time: ", np.max(timestamp_arr))
        timestamp_arr = timestamp_arr - least_time
        timestamp_arr = timestamp_arr // (interval_hour * 3600)
        return timestamp_arr


if __name__ == '__main__':
    edgelist_dataset = EdgeListData("dataset/yelp_small/train.txt", "dataset/yelp_small/test.txt")
