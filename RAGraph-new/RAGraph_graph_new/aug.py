import torch
import copy
import random
import pdb
import scipy.sparse as sp
import numpy as np

def main():
    pass


def aug_random_mask(input_feature, drop_percent=0.2):

    node_num = input_feature.shape[1]
    mask_num = int(node_num * drop_percent)
    node_idx = [i for i in range(node_num)]
    mask_idx = random.sample(node_idx, mask_num)
    aug_feature = copy.deepcopy(input_feature)
    zeros = torch.zeros_like(aug_feature[0][0])
    for j in mask_idx:
        aug_feature[0][j] = zeros
    return aug_feature


def aug_random_edge(input_adj, drop_percent=0.2):
    # === 保证输入是 dense ===
    if sp.issparse(input_adj):
        input_adj = input_adj.todense()
    else:
        input_adj = np.array(input_adj)  # 保证是 ndarray

    percent = drop_percent / 2

    row_idx, col_idx = np.nonzero(input_adj)

    edge_set = set()
    for u, v in zip(row_idx, col_idx):
        if u < v:
            edge_set.add((u, v))

    edge_list = list(edge_set)
    edge_num = len(edge_list)
    add_drop_num = int(edge_num * percent)

    drop_idx = random.sample(range(edge_num), min(add_drop_num, edge_num))
    retained_edges = [e for i, e in enumerate(edge_list) if i not in drop_idx]

    aug_adj = copy.deepcopy(input_adj.tolist())

    for u, v in edge_list:
        aug_adj[u][v] = 0
        aug_adj[v][u] = 0
    for u, v in retained_edges:
        aug_adj[u][v] = 1
        aug_adj[v][u] = 1

    node_num = input_adj.shape[0]
    existing_set = set(retained_edges)
    candidate_edges = [(i, j) for i in range(node_num) for j in range(i)
                       if (i, j) not in existing_set and (j, i) not in existing_set]

    add_list = random.sample(candidate_edges, min(add_drop_num, len(candidate_edges)))
    for u, v in add_list:
        aug_adj[u][v] = 1
        aug_adj[v][u] = 1

    aug_adj = np.array(aug_adj)  # 保证输出是 dense ndarray
    return aug_adj




def aug_drop_node(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)

    node_num = input_fea.shape[0]
    drop_num = int(node_num * drop_percent)    # number of drop nodes
    all_node_list = [i for i in range(node_num)]

    drop_node_list = sorted(random.sample(all_node_list, drop_num))

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj


def aug_subgraph(input_fea, input_adj, drop_percent=0.2):

    input_adj = torch.tensor(input_adj.todense().tolist())
    input_fea = input_fea.squeeze(0)
    node_num = input_fea.shape[0]

    all_node_list = [i for i in range(node_num)]
    s_node_num = int(node_num * (1 - drop_percent))
    center_node_id = random.randint(0, node_num - 1)
    sub_node_id_list = [center_node_id]
    all_neighbor_list = []

    for i in range(s_node_num - 1):

        all_neighbor_list += torch.nonzero(input_adj[sub_node_id_list[i]], as_tuple=False).squeeze(1).tolist()

        all_neighbor_list = list(set(all_neighbor_list))
        new_neighbor_list = [n for n in all_neighbor_list if not n in sub_node_id_list]
        if len(new_neighbor_list) != 0:
            new_node = random.sample(new_neighbor_list, 1)[0]
            sub_node_id_list.append(new_node)
        else:
            break


    drop_node_list = sorted([i for i in all_node_list if not i in sub_node_id_list])

    aug_input_fea = delete_row_col(input_fea, drop_node_list, only_row=True)
    aug_input_adj = delete_row_col(input_adj, drop_node_list)

    aug_input_fea = aug_input_fea.unsqueeze(0)
    aug_input_adj = sp.csr_matrix(np.matrix(aug_input_adj))

    return aug_input_fea, aug_input_adj





def delete_row_col(input_matrix, drop_list, only_row=False):

    remain_list = [i for i in range(input_matrix.shape[0]) if i not in drop_list]
    out = input_matrix[remain_list, :]
    if only_row:
        return out
    out = out[:, remain_list]

    return out



















if __name__ == "__main__":
    main()

