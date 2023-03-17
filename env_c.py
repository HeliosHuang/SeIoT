import dgl.data
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers_for_pre import GraphAttentionLayer
import dgl
from models_GAT import GAT
import random
import numpy as np
from copy import deepcopy
import os
import sys

seed = 2020
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
dev_vs_env = pd.read_csv('./members_env_c.csv', index_col=[0, 1])
heater_1 = dev_vs_env['heater_1'].values.reshape(-1,1)
heater_2 = dev_vs_env['heater_2'].values.reshape(-1,1)
humidifier_mi_1 = dev_vs_env['humidifier_mi_1'].values.reshape(-1,1)
humidifier_mi_2 = dev_vs_env['humidifier_mi_2'].values.reshape(-1,1)
AC = dev_vs_env['AC'].values.reshape(-1,1)
dehumidifier = dev_vs_env['dehumidifier'].values.reshape(-1,1)
humidifier_der = dev_vs_env['humidifier_der'].values.reshape(-1,1)

file_list = os.listdir('state_record')

env_fea_list = []
dev_action_list = []
for file_name in file_list:
    dev_fea = pd.read_csv('./state_record/'+file_name, index_col=[0, 1])
    dev_fea = dev_fea.to_numpy().transpose()
    fea_filter = []
    for num, record_ in enumerate(dev_fea):
        if num < 6:
            continue
        if dev_fea[num][14] != dev_fea[num-1][14]:
            fea_filter.append(record_)
    dev_fea = np.array(fea_filter)
    dev_fea = dev_fea.reshape(-1,19,1)
    dev_action = dev_fea[:, (1,2,3,4,11,12,13), :]

    env_fea = deepcopy(dev_fea[:, 14:17, :])
    env_ = []
    for env in env_fea:
        env_.append([(env[0][0] - 20) / 5, (env[1][0] - 20) / 5, (env[2][0] - 0.4) * 5])
    env_ = torch.FloatTensor(env_)
    env_fea = env_.reshape(-1, 3, 1)
    dev_action_list.append(torch.FloatTensor(dev_action))
    env_fea_list.append(torch.FloatTensor(env_fea))


hg = dgl.heterograph({
    ('env', 'eh', 'heater_1'): heater_1.nonzero(),
    ('heater_1', 'he', 'env'): heater_1.transpose().nonzero(),
    ('env', 'es', 'heater_2'): heater_2.nonzero(),
    ('heater_2', 'se', 'env'): heater_2.transpose().nonzero(),
    ('env', 'ea', 'humidifier_mi_1'): humidifier_mi_1.nonzero(),
    ('humidifier_mi_1', 'ae', 'env'): humidifier_mi_1.transpose().nonzero(),
    ('env', 'em', 'humidifier_mi_2'): humidifier_mi_2.nonzero(),
    ('humidifier_mi_2', 'me', 'env'): humidifier_mi_2.transpose().nonzero(),
    ('env', 'eac', 'AC'): AC.nonzero(),
    ('AC', 'ace', 'env'): AC.transpose().nonzero(),
    ('env', 'ed', 'dehumidifier'): dehumidifier.nonzero(),
    ('dehumidifier', 'de', 'env'): dehumidifier.transpose().nonzero(),
    ('env', 'eder', 'humidifier_der'): humidifier_der.nonzero(),
    ('humidifier_der', 'dere', 'env'): humidifier_der.transpose().nonzero(),
})
print(hg)


class GAT(nn.Module):
    def __init__(self, nfeat, nhid, nheads, dropout=0.2, alpha=0.2,):
        """Dense version of GAT."""
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
                self._cached_coalesced_graph[meta_path] = dgl.metapath_reachable_graph(
                        g, meta_path)

        for i, meta_path in enumerate(self.meta_paths):
            h_ = torch.cat((h, e[i].expand(h.shape[0], 1)), 1)
            new_g = self._cached_coalesced_graph[meta_path]
            adj_ = new_g.adj().to_dense()
            semantic_embeddings.append(self.gat_layers[i](h_, adj_).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)
        return self.semantic_attention(semantic_embeddings)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout):
        super(HAN, self).__init__()
        self.dropout = dropout
        self.embed = nn.Embedding(3, 2, max_norm=1)
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l-1],
                                        hidden_size, num_heads[l], dropout))
        self.predict = nn.Linear(16, out_size)
        self.dim2 = hidden_size * num_heads[-1]
        self.decode1 = nn.Sequential(
            nn.Linear(256,128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
        )
        self.decode2 = nn.Sequential(
            nn.Linear(256,128),
            nn.ELU(inplace=True),
            nn.Linear(128, 64),
            nn.ELU(inplace=True),
            nn.Linear(64, 32),
            nn.ELU(inplace=True),
            nn.Linear(32, 16),
            nn.ELU(inplace=True),
        )
        self.decode3 = nn.Sequential(
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
        num_ = torch.LongTensor(range(3)).reshape(-1,1)
        x_ind = self.embed(num_).reshape(3,-1)
        h = torch.cat((x_ind, h), 1)
        for gnn in self.layers:
            h = gnn(g, h, e)
        h0 = self.decode1(h[0].reshape(1,-1))
        h0 = F.dropout(h0, self.dropout, training=self.training)
        h1 = self.decode2(h[1].reshape(1,-1))
        h1 = F.dropout(h1, self.dropout, training=self.training)
        h2 = self.decode3(h[2].reshape(1,-1))
        h2 = F.dropout(h2, self.dropout, training=self.training)
        h = torch.cat((h0, h1, h2), 0)
        h = self.predict(h)
        return h


if __name__ == "__main__":
    if sys.argv[1] == 't':
        model = HAN(meta_paths=[['eh', 'he'], ['es', 'se'], ['ea', 'ae'], ['em', 'me'], ['eac', 'ace'], ['ed', 'de'], ['eder', 'dere']],
                    in_size=4,
                    hidden_size=32,
                    out_size=1,
                    num_heads=[8],
                    dropout=0.01)
        embed_para = list(map(id, model.embed.parameters()))
        other_params = filter(lambda p: id(p) not in embed_para, model.parameters())
        lr = 0.00002
        params = [{'params': other_params},
                  {'params': model.embed.parameters(), 'lr': lr * 50}]
        optimizer = torch.optim.Adam(params, lr=lr)
        loss_fcn = torch.nn.MSELoss(reduction='mean')

        for epoch in range(100):
            model.train()
            for file_num in range(1,7):
                dev_action_1 = dev_action_list[file_num]
                env_fea_1 = env_fea_list[file_num]
                for i, feature_ in enumerate(env_fea_1[0:]):
                    if i == 0:
                        former_feature_1 = feature_
                        continue
                    if i == len(env_fea_1) - 2:
                        break
                    # feature = torch.cat((former_feature), 1)
                    feature = deepcopy(former_feature_1)
                    feature = feature.reshape(3, -1)
                    former_feature_1 = feature_
                    pred = model(hg, feature, dev_action_1[i-1:i].permute(1,0,2).reshape(7,1))
                    # loss = loss_fcn(pred, labels.reshape(-1))
                    loss = loss_fcn(pred, env_fea_1[i+0])
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            model.eval()
            print('epoch:{} loss:{:7f}'.format(epoch, loss.item()))
            temp1_error = []
            temp2_error = []
            humidity_error = []
            for file_num in range(7,14):
                dev_action_1 = dev_action_list[file_num]
                env_fea_1 = env_fea_list[file_num]
                for i, feature_ in enumerate(env_fea_1):
                    if i == 0:
                        former_feature_1 = feature_
                        continue
                    if i == len(env_fea_1) - 2:
                        break
                    # feature = torch.cat((former_feature), 1)
                    feature = deepcopy(former_feature_1)
                    feature = feature.reshape(3, -1)
                    former_feature_1 = feature_
                    pred = model(hg, feature, dev_action_1[i-1:i].permute(1,0,2).reshape(7,1))

                    temp1_error.append(abs(pred[0].detach().numpy()[0] - np.array(env_fea_1[i+0])[0][0]))
                    temp2_error.append(abs(pred[1].detach().numpy()[0] - np.array(env_fea_1[i+0])[1][0]))
                    humidity_error.append(abs(pred[2].detach().numpy()[0] - np.array(env_fea_1[i+0])[2][0]))
            print('temp error:')
            print((np.mean(temp1_error) + np.mean(temp2_error))/2)
            print('humidity error:')
            print(np.mean(humidity_error))
        torch.save(model, 'the_model_for_env_c')
    if sys.argv[1] == 'te':
        model = torch.load('pre_trained_model/env_c_model')
        model.eval()
        temp1_error = []
        temp2_error = []
        humidity_error = []
        for file_num in range(7,14):
            dev_action_1 = dev_action_list[file_num]
            env_fea_1 = env_fea_list[file_num]
            for i, feature_ in enumerate(env_fea_1):
                if i == 0:
                    former_feature_1 = feature_
                    continue
                if i == len(env_fea_1) - 2:
                    break
                # feature = torch.cat((former_feature), 1)
                feature = deepcopy(former_feature_1)
                feature = feature.reshape(3, -1)
                former_feature_1 = feature_
                pred = model(hg, feature, dev_action_1[i-1:i].permute(1,0,2).reshape(7,1))

                temp1_error.append(abs(pred[0].detach().numpy()[0] - np.array(env_fea_1[i+0])[0][0]))
                temp2_error.append(abs(pred[1].detach().numpy()[0] - np.array(env_fea_1[i+0])[1][0]))
                humidity_error.append(abs(pred[2].detach().numpy()[0] - np.array(env_fea_1[i+0])[2][0]))
        print('temp error:')
        print((np.mean(temp1_error) + np.mean(temp2_error)) * 5 / 2)
        print('humidity error:')
        print(np.mean(humidity_error) / 5)
