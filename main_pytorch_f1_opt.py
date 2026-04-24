import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from numpy import random
from torch.nn.parameter import Parameter
import os
import pickle

from homhetgnn.HGB.LP.benchmark.methods.GATNE.utils_f1_opt import *
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
sys.path.append('../../')
from scripts.data_loader import data_loader

def get_batches(pairs, neighbors, batch_size):
    n_batches = (len(pairs) + (batch_size - 1)) // batch_size

    for idx in range(n_batches):
        x, y, t, neigh = [], [], [], []
        for i in range(batch_size):
            index = idx * batch_size + i
            if index >= len(pairs):
                break
            x.append(pairs[index][0])
            y.append(pairs[index][1])
            t.append(pairs[index][2])
            neigh.append(neighbors[pairs[index][0]])
        yield torch.tensor(x), torch.tensor(y), torch.tensor(t), torch.tensor(neigh)


class GATNEModel(nn.Module):
    def __init__(
        self, num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features, dropout=0.2
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.embedding_size = embedding_size
        self.embedding_u_size = embedding_u_size
        self.edge_type_count = edge_type_count
        self.dim_a = dim_a
        self.features = features

        if features is not None:
            feature_dim = features.shape[-1]
            self.embed_trans = Parameter(torch.empty(feature_dim, embedding_size))
            self.u_embed_trans = Parameter(torch.empty(edge_type_count, feature_dim, embedding_u_size))
        else:
            self.node_embeddings = Parameter(torch.empty(num_nodes, embedding_size))
            self.node_type_embeddings = Parameter(
                torch.empty(num_nodes, edge_type_count, embedding_u_size)
            )

        self.trans_weights = Parameter(
            torch.empty(edge_type_count, embedding_u_size, embedding_size)
        )

        self.type_att_s1 = nn.Linear(embedding_u_size, dim_a)
        self.type_att_s2 = nn.Linear(dim_a, 1)

        self.neigh_att_s1 = nn.Linear(embedding_u_size, dim_a)
        self.neigh_att_s2 = nn.Linear(dim_a, 1)

        self.gate = nn.Linear(2 * embedding_size, embedding_size)
        self.norm = nn.LayerNorm(embedding_size)
        self.dropout = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        if self.features is not None:
            nn.init.xavier_uniform_(self.embed_trans)
            nn.init.xavier_uniform_(self.u_embed_trans)
        else:
            nn.init.xavier_uniform_(self.node_embeddings)
            nn.init.xavier_uniform_(self.node_type_embeddings)

        nn.init.xavier_uniform_(self.trans_weights)

        for layer in [self.type_att_s1, self.type_att_s2, self.neigh_att_s1, self.neigh_att_s2, self.gate]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, train_inputs, train_types, node_neigh):
        if self.features is None:
            node_embed = self.node_embeddings[train_inputs]  # [B, D]
            node_embed_neighbors = self.node_type_embeddings[node_neigh]
        else:
            node_embed = self.features[train_inputs] @ self.embed_trans
            node_embed_neighbors = torch.einsum(
                'bijk,akm->bijam', self.features[node_neigh], self.u_embed_trans
            )

        node_embed_tmp = torch.cat(
            [
                node_embed_neighbors[:, i, :, i, :].unsqueeze(1)
                for i in range(self.edge_type_count)
            ],
            dim=1,
        )  # [B, T, N, Du]

        # Attention over neighbors within each type
        neigh_scores = self.neigh_att_s2(
            torch.tanh(self.neigh_att_s1(node_embed_tmp))
        ).squeeze(-1)  # [B, T, N]
        neigh_alpha = F.softmax(neigh_scores, dim=2).unsqueeze(-1)
        node_type_embed = torch.sum(neigh_alpha * node_embed_tmp, dim=2)  # [B, T, Du]

        # Attention over edge types
        att_scores = self.type_att_s2(
            torch.tanh(self.type_att_s1(node_type_embed))
        ).squeeze(-1)  # [B, T]
        att = F.softmax(att_scores, dim=1).unsqueeze(1)  # [B, 1, T]
        node_type_embed = torch.matmul(att, node_type_embed)  # [B, 1, Du]

        trans_w = self.trans_weights[train_types]  # [B, Du, D]
        type_msg = torch.matmul(node_type_embed, trans_w).squeeze(1)  # [B, D]
        type_msg = self.dropout(type_msg)

        g = torch.sigmoid(self.gate(torch.cat([node_embed, type_msg], dim=1)))
        node_embed = node_embed + g * type_msg
        node_embed = self.norm(node_embed)

        return F.normalize(node_embed, dim=1)


class NSLoss(nn.Module):
    def __init__(self, num_nodes, num_sampled, embedding_size):
        super(NSLoss, self).__init__()
        self.num_nodes = num_nodes
        self.num_sampled = num_sampled
        self.embedding_size = embedding_size
        self.weights = Parameter(torch.FloatTensor(num_nodes, embedding_size))
        self.sample_weights = F.normalize(
            torch.Tensor(
                [
                    (math.log(k + 2) - math.log(k + 1)) / math.log(num_nodes + 1)
                    for k in range(num_nodes)
                ]
            ),
            dim=0,
        )

        self.reset_parameters()

    def reset_parameters(self):
        self.weights.data.normal_(std=1.0 / math.sqrt(self.embedding_size))

    def forward(self, input, embs, label):
        n = input.shape[0]
        log_target = torch.log(
            torch.sigmoid(torch.sum(torch.mul(embs, self.weights[label]), 1))
        )
        negs = torch.multinomial(
            self.sample_weights, self.num_sampled * n, replacement=True
        ).view(n, self.num_sampled)
        noise = torch.neg(self.weights[negs])
        sum_log_sampled = torch.sum(
            torch.log(torch.sigmoid(torch.bmm(noise, embs.unsqueeze(2)))), 1
        ).squeeze()

        loss = log_target + sum_log_sampled
        return -loss.sum() / n


def train_model(network_data, feature_dic):
    pickle_dir = './pickles'
    #print current directory
    print(f'Current directory: {os.getcwd()}')
    if not os.path.exists(pickle_dir):
        os.makedirs(pickle_dir, exist_ok=True)
    walk_file = os.path.join(pickle_dir,f'{args.data}-walks-pickle')
    pair_file = os.path.join(pickle_dir,f'{args.data}-pair-pickle')
    vocab, index2word, train_pairs = generate(network_data, args.num_walks, args.walk_length, args.schema,
                                              args.window_size, args.num_workers, walk_file=walk_file,
                                              pair_file=pair_file, node_type=dl.types['data'])

    edge_types = list(network_data.keys())

    num_nodes = len(index2word)
    edge_type_count = len(edge_types)
    epochs = args.epoch
    batch_size = args.batch_size
    embedding_size = args.dimensions
    embedding_u_size = args.edge_dim
    u_num = edge_type_count
    num_sampled = args.negative_samples
    dim_a = args.att_dim
    att_head = 1
    neighbor_samples = args.neighbor_samples

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Info: device: {device}')
    neigh_file = f'pickles/{args.data}-neighs-pickle'
    # if os.path.exists(neigh_file):
    #     print(f'Data: Load neighs from {neigh_file}')
    #     neighbors = pickle.load(open(neigh_file, 'rb'))
    # else:
    neighbors = generate_neighbors(network_data, vocab, num_nodes, edge_types, neighbor_samples)
    pickle.dump(neighbors, open(neigh_file, 'wb'))

    features = None
    if feature_dic is not None:
        feature_dim = len(list(feature_dic.values())[0])
        print('feature dimension: ' + str(feature_dim))
        features = np.zeros((num_nodes, feature_dim), dtype=np.float32)
        for key, value in feature_dic.items():
            if key in vocab:
                features[vocab[key].index, :] = np.array(value)
        features = torch.FloatTensor(features).to(device)

    model = GATNEModel(
        num_nodes, embedding_size, embedding_u_size, edge_type_count, dim_a, features
    )
    nsloss = NSLoss(num_nodes, num_sampled, embedding_size)

    model.to(device)
    nsloss.to(device)

    optimizer = torch.optim.Adam(
        [{"params": model.parameters()}, {"params": nsloss.parameters()}], lr=1e-3
    )

    best_score = 0
    patience = 0
    test_2hop_best, test_random_best = None, None
    for epoch in range(epochs):
        random.shuffle(train_pairs)
        batches = get_batches(train_pairs, neighbors, batch_size)

        data_iter = tqdm(
            batches,
            desc="epoch %d" % (epoch),
            total=(len(train_pairs) + (batch_size - 1)) // batch_size,
            bar_format="{l_bar}{r_bar}",
        )
        avg_loss = 0.0

        for i, data in enumerate(data_iter):
            optimizer.zero_grad()
            embs = model(data[0].to(device), data[2].to(device), data[3].to(device), )
            loss = nsloss(data[0].to(device), embs, data[1].to(device))
            loss.backward()
            optimizer.step()

            avg_loss += loss.item()

            if i % 10000 == 0:
                post_fix = {
                    "epoch": epoch,
                    "iter": i,
                    "avg_loss": avg_loss / (i + 1),
                    "loss": loss.item(),
                }
                data_iter.write(str(post_fix))
            # break
        final_model = dict(zip(edge_types, [dict() for _ in range(edge_type_count)]))
        for i in range(num_nodes):
            train_inputs = torch.tensor([i for _ in range(edge_type_count)], dtype=torch.int64).to(device)
            train_types = torch.tensor(list(range(edge_type_count)), dtype=torch.int64).to(device)
            node_neigh = torch.tensor(
                [neighbors[i] for _ in range(edge_type_count)], dtype=torch.int64
            ).to(device)
            node_emb = model(train_inputs, train_types, node_neigh)
            for j in range(edge_type_count):
                final_model[edge_types[j]][index2word[i]] = (
                    node_emb[j].cpu().detach().numpy()
                )

        valid_scores, test_2hop_scores, test_random_scores = defaultdict(list), defaultdict(list), defaultdict(list)
        for i in range(edge_type_count):
            if args.eval_type == "all" or edge_types[i] in args.eval_type.split(","):
                for etype in edge_types:
                    print(
                        etype,
                        "valid:", len(valid_true_data_by_edge.get(etype, [])), len(valid_false_data_by_edge.get(etype, [])),
                        "test:", len(testing_true_data_by_edge.get(etype, [])), len(testing_false_data_by_edge.get(etype, [])),
                        "random:", len(random_testing_true_data_by_edge.get(etype, [])), len(random_testing_false_data_by_edge.get(etype, [])),
                    )
                valid_score = evaluate(
                    final_model[edge_types[i]],
                    valid_true_data_by_edge[edge_types[i]],
                    valid_false_data_by_edge[edge_types[i]],
                    dl
                )
                for k in valid_score.keys():
                    valid_scores[k].append(valid_score[k])
                test_2hop_score = evaluate(
                    final_model[edge_types[i]],
                    testing_true_data_by_edge[edge_types[i]],
                    testing_false_data_by_edge[edge_types[i]],
                    dl
                )
                for k in test_2hop_score.keys():
                    test_2hop_scores[k].append(test_2hop_score[k])

                test_random_score = evaluate(
                    final_model[edge_types[i]],
                    random_testing_true_data_by_edge[edge_types[i]],
                    random_testing_false_data_by_edge[edge_types[i]],
                    dl
                )
                for k in test_random_score.keys():
                    test_random_scores[k].append(test_random_score[k])
        valid_score_mean = dict()
        for k in valid_scores.keys():
            valid_score_mean[k] = np.mean(valid_scores[k])
        print(f"valid score: {valid_score_mean}")

        test_2hop_score_mean = dict()
        for s in test_2hop_scores.keys():
            test_2hop_score_mean[s] = np.mean(test_2hop_scores[s])
        print(f"test 2hop scroe: {test_2hop_score_mean}")

        test_random_score_mean = dict()
        for s in test_random_scores.keys():
            test_random_score_mean[s] = np.mean(test_random_scores[s])
        print(f"test random score: {test_random_score_mean}")

        cur_score = valid_score_mean['roc_auc']
        if cur_score > best_score:
            best_score = cur_score
            test_2hop_best, test_random_best = test_2hop_scores, test_random_scores
            patience = 0
        else:
            patience += 1
            print(f'patience is {patience} of total {args.patience}')
            if patience > args.patience:
                print("Early Stopping")
                break
    return test_2hop_best, test_random_best


if __name__ == "__main__":
    args = parse_args()
    print('args: ', args)
    file_name = 'pickles'
    if args.features is not None:
        feature_dic = load_feature_data(args.features)
    else:
        feature_dic = None
    #print features keys and length and len(values)
    if feature_dic is not None:
        lengths = [len(v) for v in feature_dic.values()]
        print(f"Info: feature_dic length (num nodes): {len(feature_dic)}")
        print(f"Info: keys sample: {list(feature_dic.keys())[:5]}")
        print(f"Info: sample embedding[0] (len {lengths[0]}): {list(feature_dic.values())[0][:10]} ...")
        print(f"Info: sample embedding[1] (len {lengths[1]}): {list(feature_dic.values())[1][:10]} ...")
        print(f"Info: min feature length: {min(lengths)}, max feature length: {max(lengths)}")

    data_name = args.data
    data_dir = os.path.join('../../data', data_name)
    node_type_file = os.path.join('../../data', data_name, 'node.dat')
    dl_pickle_f = os.path.join(data_dir, 'dl_pickle')
    # if os.path.exists(dl_pickle_f):
    #     dl = pickle.load(open(dl_pickle_f, 'rb'))
    #     print(f'Info: load {data_name} from {dl_pickle_f}')
    # else:
    dl = data_loader(data_dir)
    pickle.dump(dl, open(dl_pickle_f, 'wb'))
    print(f'Info: load {data_name} from original data and generate {dl_pickle_f} ')

    """ Load train/valid/test data """
    training_data_by_type = load_train_data(dl)
    valid_true_data_by_edge, valid_false_data_by_edge = load_valid_data(dl)
    testing_true_data_by_edge, testing_false_data_by_edge = load_test_data(dl, random_test=False)
    random_testing_true_data_by_edge, random_testing_false_data_by_edge = load_test_data(dl, random_test=True)


    test_2hop_best, test_random_best = train_model(training_data_by_type, feature_dic)

    if test_2hop_best is None or test_random_best is None:
        print("No best model was stored.")
    else:
        for k in test_2hop_best.keys():
            test_2hop_best[k] = np.around(np.mean(test_2hop_best[k]), 4)
            test_random_best[k] = np.around(np.mean(test_random_best[k]), 4)

        print("\nFinal best results")
        print(
            f"Test 2hop -> "
            f"roc_auc: {test_2hop_best.get('roc_auc', 'N/A')}, "
            f"hit@1: {test_2hop_best.get('hit@1', 'N/A')}, "
            f"precision: {test_2hop_best.get('precision', 'N/A')}, "
            f"recall: {test_2hop_best.get('recall', 'N/A')}, "
            f"f1: {test_2hop_best.get('f1', 'N/A')}"
        )
        print(
            f"Test random -> "
            f"roc_auc: {test_random_best.get('roc_auc', 'N/A')}, "
            f"hit@1: {test_random_best.get('hit@1', 'N/A')}, "
            f"precision: {test_random_best.get('precision', 'N/A')}, "
            f"recall: {test_random_best.get('recall', 'N/A')}, "
            f"f1: {test_random_best.get('f1', 'N/A')}"
        )
    
