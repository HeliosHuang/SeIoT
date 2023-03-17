import dgl.data
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers import GraphAttentionLayer
import dgl
from dgl.nn.pytorch import GATConv
from dgl.nn.pytorch import SAGEConv
from models_GAT import GAT
import random
import numpy as np
from copy import deepcopy
import os
import sys


def prepare_data(feature_file):
    dev_fea = pd.read_csv('./state_record/'+feature_file, index_col=[0, 1])
    dev_fea = dev_fea.to_numpy().transpose()
    dev_fea = dev_fea.reshape(-1, 19, 1)
    dev_action = dev_fea[:, 0:14, :]
    dev_label = dev_action
    env_fea = deepcopy(dev_fea[:, 14:, :])
    dev_action = torch.FloatTensor(dev_action)
    env_fea = torch.FloatTensor(env_fea)
    env_ = []
    for env in env_fea:
        env_.append([(env[0][0] - 20)*1.5, (30 - env[1][0])*1.5, env[2][0], env[3][0], env[4][0]])
    env_ = torch.FloatTensor(env_)
    env_fea = env_.reshape(-1, 5, 1)
    return dev_action, env_fea, dev_label


seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# obtain adjacency matrix
dev_vs_env = pd.read_csv('./members.csv', index_col=[0, 1])
temperature1_env = dev_vs_env['temperature1'].values.reshape(-1,1)
temperature2_env = dev_vs_env['temperature2'].values.reshape(-1,1)
humidity_env = dev_vs_env['humidity'].values.reshape(-1,1)
motion_env = dev_vs_env['motion'].values.reshape(-1,1)
light_env = dev_vs_env['light'].values.reshape(-1,1)

# obtain node feature (train: day1-day7, test: day8-day14 )
dev_action_list = []
env_fea_list = []
dev_label_list = []
file_list = os.listdir('state_record')

for filename in file_list:
    dev_action, env_fea, dev_label = prepare_data(filename)
    dev_action_list.append(deepcopy(dev_action))
    env_fea_list.append(deepcopy(env_fea))
    dev_label_list.append(deepcopy(dev_label))


hg = dgl.heterograph({
    ('temperature', 'td', 'device'): temperature1_env.transpose().nonzero(),
    ('device', 'dt', 'temperature'): temperature1_env.nonzero(),
    ('temperature2', 't2d', 'device'): temperature2_env.transpose().nonzero(),
    ('device', 'dt2', 'temperature2'): temperature2_env.nonzero(),
    ('humidity', 'hd', 'device'): humidity_env.transpose().nonzero(),
    ('device', 'dh', 'humidity'): humidity_env.nonzero(),
    ('light', 'ld', 'device'): light_env.transpose().nonzero(),
    ('device', 'dl', 'light'): light_env.nonzero(),
    ('motion', 'md', 'device'): motion_env.transpose().nonzero(),
    ('device', 'dm', 'motion'): motion_env.nonzero(),
})
print(hg)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout=0.2, alpha=0.2,):
        super(GAT, self).__init__()
        self.dropout = dropout
        self.attentions = [GraphAttentionLayer(nfeat, nhid, dropout=dropout, alpha=alpha, concat=True) for _ in range(nheads)]
        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, x, adj):
        x = torch.cat([att(x, adj) for att in self.attentions], dim=1)
        x = F.dropout(x, self.dropout, training=self.training)
        return x


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128):
        super(SemanticAttention, self).__init__()
        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w2 = self.project(z)
        beta = torch.softmax(w2, dim=1)
        return (beta * z).sum(1)


class HANLayer(nn.Module):
    def __init__(self, meta_paths, in_size, out_size, layer_num_heads, dropout):
        super(HANLayer, self).__init__()
        # one GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        for i in range(len(meta_paths)):
            self.gat_layers.append(GAT(in_size, out_size, layer_num_heads, dropout=dropout))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads)
        self.meta_paths = list(tuple(meta_path) for meta_path in meta_paths)

        self._cached_graph = None
        self._cached_coalesced_graph = {}

    def forward(self, g, h, e):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_path in self.meta_paths:
                # meta-path based sub graph extraction
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            # meta-path-based feature added
            h_ = torch.cat((h, e[i].expand(h.shape[0], 2)), 1)
            new_g = self._cached_coalesced_graph[meta_path]
            adj_ = new_g.adj().to_dense()
            semantic_embeddings.append(self.gat_layers[i](h_, adj_).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout, device_num=14):
        super(HAN, self).__init__()
        self.dropout = dropout
        self.embed = nn.Embedding(device_num, 2, max_norm=1)  # ID embedded layer
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(16, out_size)
        self.dim2 = hidden_size * num_heads[-1]
        self.decode = nn.Sequential(
            nn.Linear(256,128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
        )
        self.OutLayer = nn.Softmax(dim=1)

    def forward(self, g, h, e):
        num_ = torch.LongTensor(range(14)).reshape(-1,1)  # generate ID
        x_ind = self.embed(num_).reshape(14,-1)  # ID embedding
        h = torch.cat((x_ind, h), 1)
        for gnn in self.layers:
            h = gnn(g, h, e)
        h = self.decode(h)
        h = F.dropout(h, self.dropout, training=self.training)
        h = self.predict(h)
        return h


if __name__ == "__main__":
    if sys.argv[1] == 't':
        model = HAN(meta_paths=[['dt', 'td'], ['dt2', 't2d'], ['dh', 'hd'],  ['dl', 'ld'], ['dm', 'md']],
                    in_size=8,
                    hidden_size=32,
                    out_size=2,
                    num_heads=[8],
                    dropout=0.01,)
        embed_para = list(map(id, model.embed.parameters()))
        other_params = filter(lambda p: id(p) not in embed_para, model.parameters())
        lr = 0.0002
        params = [{'params': other_params},
                  {'params': model.embed.parameters(), 'lr': lr * 50}]
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fcn = torch.nn.NLLLoss()
        for epoch in range(20):
            model.train()
            for file_num in range(7):
                print('now train file number: ' + str(file_num+1) + '/' + str(7))
                dev_action = dev_action_list[file_num]
                env_fea = env_fea_list[file_num]
                dev_label = dev_label_list[file_num]
                for i, feature_ in enumerate(dev_action):
                    if i == 0:
                        former_feature = feature_
                        continue
                    feature = torch.cat((former_feature, feature_), 1)
                    feature = feature.reshape(14, -1)  # 14 is device number
                    former_feature = feature_
                    new_feature = []
                    # 1-hot for device state
                    for dev_fea in feature:
                        if dev_fea[0] == 0 and dev_fea[1] == 0:
                            new_feature.append([1, 0, 1, 0])
                        elif dev_fea[0] == 0 and dev_fea[1] == 1:
                            new_feature.append([1, 0, 0, 1])
                        elif dev_fea[0] == 1 and dev_fea[1] == 0:
                            new_feature.append([0, 1, 0, 0])
                        elif dev_fea[0] == 1 and dev_fea[1] == 1:
                            new_feature.append([0, 1, 0, 1])
                    new_feature = torch.LongTensor(new_feature).reshape(-1, 4)
                    pred = model(hg, new_feature, env_fea[i-1:i+1].permute(1,0,2).reshape(5,2))
                    labels = torch.tensor(dev_label[i], dtype=torch.long)
                    loss = F.cross_entropy(pred, labels.reshape(-1))
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))
            model.eval()
            num_correct, num_wrong = 0, 0
            print('start_test')
            for file_num in range(7,14):
                dev_action = dev_action_list[file_num]
                env_fea = env_fea_list[file_num]
                dev_label = dev_label_list[file_num]
                for i, feature_ in enumerate(dev_action):
                    if i == 0:
                        former_feature = feature_
                        continue
                    feature = torch.cat((former_feature, feature_), 1)
                    feature = feature.reshape(14, -1)
                    former_feature = feature_
                    new_feature = []
                    for dev_fea in feature:
                        if dev_fea[0] == 0 and dev_fea[1] == 0:
                            new_feature.append([1,0,1,0])
                        elif dev_fea[0] == 0 and dev_fea[1] == 1:
                            new_feature.append([1,0,0,1])
                        elif dev_fea[0] == 1 and dev_fea[1] == 0:
                            new_feature.append([0,1,0,0])
                        elif dev_fea[0] == 1 and dev_fea[1] == 1:
                            new_feature.append([0,1,0,1])
                    new_feature = torch.LongTensor(new_feature).reshape(-1, 4)
                    pred = model(hg, new_feature, env_fea[i-1:i+1].permute(1,0,2).reshape(5,2))
                    for num, dev_ in enumerate(pred):
                        if dev_.detach().numpy()[0] >= dev_.detach().numpy()[1] and np.array(feature_)[num][0] == 0:
                            num_correct += 1
                        elif dev_.detach().numpy()[0] < dev_.detach().numpy()[1] and np.array(feature_)[num][0] == 1:
                            num_correct += 1
                        else:
                            num_wrong += 1
            print('test fp:')
            print(num_wrong/(num_correct + num_wrong))
            torch.save(model, 'the_model_'+str(epoch))

    if sys.argv[1] == 'te':
        model = torch.load('pre_trained_model/node_c_model')
        model.eval()
        dev_dic = {}
        num_correct, num_wrong, num_test = 0, 0, 0
        for file_num in range(7,14):
            print('now test file number: ' + str(file_num-6) + '/7')
            dev_action = dev_action_list[file_num]
            env_fea = env_fea_list[file_num]
            dev_label = dev_label_list[file_num]
            for i, feature_ in enumerate(dev_action):
                if i == 0:
                    former_feature = feature_
                    continue
                feature = torch.cat((former_feature, feature_), 1)
                feature = feature.reshape(14, -1)
                former_feature = feature_
                new_feature = []
                for dev_fea in feature:
                    if dev_fea[0] == 0 and dev_fea[1] == 0:
                        new_feature.append([1, 0, 1, 0])
                    elif dev_fea[0] == 0 and dev_fea[1] == 1:
                        new_feature.append([1, 0, 0, 1])
                    elif dev_fea[0] == 1 and dev_fea[1] == 0:
                        new_feature.append([0, 1, 0, 0])
                    elif dev_fea[0] == 1 and dev_fea[1] == 1:
                        new_feature.append([0, 1, 0, 1])
                new_feature = torch.LongTensor(new_feature).reshape(-1, 4)
                pred = model(hg, new_feature, env_fea[i-1:i+1].permute(1,0,2).reshape(5,2))
                flag = 0
                for num, dev_ in enumerate(pred):
                    if dev_.detach().numpy()[0] >= dev_.detach().numpy()[1] and np.array(feature_)[num][0] == 0:
                        pass
                    elif dev_.detach().numpy()[0] < dev_.detach().numpy()[1] and np.array(feature_)[num][0] == 1:
                        pass
                    else:
                        if num in dev_dic:
                            dev_dic[num] += 1
                        else:
                            dev_dic[num] = 1
                        flag = 1
                if flag == 1:
                    num_wrong += 1
                else:
                    num_correct += 1
        print('TNR: ' + str(num_correct/(num_correct+num_wrong)))
        print(dev_dic)