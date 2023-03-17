import pandas as pd
import sys
import os
import numpy as np
import json
from copy import deepcopy
import math
import pickle
import csv
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
import random
feature_dir = 'feature_idle'


def _str2tuple(key):
    a = []
    back = 0
    for num, i in enumerate(key):
        if i == ',' or i == ')':
            fore = back + 1
            back = num
            try:
                a.append(int(key[fore:back]))
            except:
                pass
    return tuple(a)


def rule_aggre(rule_dict):
    new_rule = {}
    for rule_num in rule_dict:
        new_rule[rule_num] = []
        for rule_ in rule_dict[rule_num]:
            if set(rule_).union(new_rule[rule_num]) != set(new_rule[rule_num]):
                new_rule[rule_num] = list(set(rule_).union(new_rule[rule_num]))

    final_list = []

    for rule_ in new_rule:
        for rule__ in final_list:
            if set(new_rule[rule_]).union(rule__) == set(rule__):
                break
        else:
            final_list.append(new_rule[rule_])
    return final_list


def edit_distance(word1, word2, mode='strict'):
    len1 = len(word1)
    len2 = len(word2)
    if len1 > 1000:
        word1 = word1[0:1000]
    if len2 > 1000:
        word2 = word2[0:1000]
    len1 = len(word1)
    len2 = len(word2)
    dp = np.zeros((len1 + 1, len2 + 1))
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j
    word1 = list(map(int, word1))
    word2 = list(map(int, word2))
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if mode == 'strict':
                delta = 1 if word1[i - 1] != word2[j - 1] else 0
            else:
                delta = 2 * abs(word1[i - 1] - word2[j - 1]) / (abs(word1[i - 1]) + abs(word2[j - 1]))
            if mode == 'strict':
                add_cost = 1
            else:
                add_cost = 0 if abs(word2[j - 1]) < 50 else 1
            if mode == 'strict':
                del_cost = 1
            else:
                del_cost = 0 if abs(word1[i - 1]) < 50 else 1
            dp[i][j] = min(dp[i - 1][j - 1] + delta, min(dp[i - 1][j] + del_cost, dp[i][j - 1] + add_cost))
    if mode == 'strict':
        return (len1 + len2 - dp[len1, len2]) / (len1 + len2)
    else:
        return (len1 + len2 - dp[len1, len2]) / (len1 + len2)


def dbscan_clu(rule_set, eps=5):
    rule_set_new = []
    # detect overlapped
    for num, rule_ in enumerate(rule_set):
        rule_union = []
        if len(rule_set_new) > 0:
            for i, rule_1 in enumerate(rule_set_new):
                for j, rule_2 in enumerate(rule_set_new):
                    if j < i:
                        continue
                    if not set(rule_1).union(rule_2) in rule_union:
                        rule_u = list(set(rule_1).union(rule_2))
                        rule_u.sort()
                        rule_union.append(rule_u)
        if rule_ in rule_union:
            pass
        else:
            rule_set_new.append(rule_)
    rule_set = rule_set_new
    new_feature = []
    for rule_1 in rule_set:
        new_feature.append([])
        for rule_2 in rule_set:
            new_feature[-1].append(round(edit_distance(rule_1,rule_2, mode='loose'),2))

    X = StandardScaler().fit_transform(new_feature)
    # Compute DBSCAN
    db = DBSCAN(eps=eps, min_samples=1).fit(X)
    labels = db.labels_
    new_class ={}
    for i in range(X.shape[0]):
        if labels[i] in new_class:
            new_class[labels[i]].append(rule_set[i])
        else:
            new_class[labels[i]] = [rule_set[i]]
    return new_class


class Burst2Iaf(object):
    def __init__(self, systeminfo, tk=None):
        self.dev_name = systeminfo
        self.trace_key = tk
        self.traffic_burst = []
        self.action_list = []
        self.burst_time = []
        self.traffic_burst_time = []
        self.traffic_set = {}

    def strict_cluster(self):
        for bur in self.traffic_burst:
            bur = list(map(int, bur))
            bur_set = set(bur)
            set_list = [i for i in bur_set]
            set_list.sort()
            if tuple(set_list) not in self.traffic_set:
                self.traffic_set[tuple(set_list)] = [1, np.mean(list(map(abs, bur))), len(bur)]
            else:
                size_mean = deepcopy((self.traffic_set[tuple(set_list)][1] * self.traffic_set[tuple(set_list)][0] +
                                     np.mean(list(map(abs, bur)))) / (self.traffic_set[tuple(set_list)][0] + 1))
                len_mean = deepcopy((self.traffic_set[tuple(set_list)][2] * self.traffic_set[tuple(set_list)][0] +
                                     len(bur)) / (self.traffic_set[tuple(set_list)][0] + 1))
                num = deepcopy(self.traffic_set[tuple(set_list)][0] + 1)
                self.traffic_set[tuple(set_list)] = [num, size_mean, len_mean]
        number_list = [self.traffic_set[key][0] for key in self.traffic_set]
        tuple_list = [list(key) for key in self.traffic_set]
        while 1:
            if len(number_list) == 0:
                break
            now_par_index = number_list.index(max(number_list))
            flag = 0
            for action_1 in self.action_list:
                for action_2 in self.action_list:
                    if action_2 == action_1:
                        continue
                    if (set(action_1).union(action_2)).difference(set(tuple_list[now_par_index])) == set() and \
                            (set(tuple_list[now_par_index])).difference(set(action_1).union(action_2)) == set():
                        flag = 1
            if flag == 0:
                self.action_list.append(tuple_list[now_par_index])
            del number_list[now_par_index]
            del tuple_list[now_par_index]

    def read_length_feature(self):
        self.traffic_burst = []
        feature_file = os.path.join(feature_dir, self.dev_name, self.trace_key + '_length.csv')
        with open(feature_file, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.traffic_burst.append(row)
        return len(self.traffic_burst) != 0

    def read_time_feature(self):
        self.traffic_burst_time = []
        feature_file = os.path.join(feature_dir, self.dev_name, self.trace_key + '_time.csv')
        with open(feature_file, 'r') as f:
            f_csv = csv.reader(f)
            for row in f_csv:
                self.traffic_burst_time.append(row)
        return len(self.traffic_burst_time) != 0

    def get_cluster_result(self):
        if not self.read_length_feature():
            print('no burst data')
            return None
        self.strict_cluster()


if __name__ == "__main__":
    dev_name = [
        'fan',
        'heater',
        'dehumidifier',
        'humidifier_der',
        'mijia_humidifier',
        'desklight',
        'bedlight',
        'ac_plug',
        'smartplug_1',
    ]
    # idle rule extractor
    fp_dir = 'fpset'
    for systeminfo in dev_name:
        print(systeminfo)
        burst_file = os.listdir(os.path.join('feature_idle', systeminfo))
        for file_name in burst_file:
            if not file_name.endswith('_length.csv'):
                continue
            feature_dir = 'feature_idle'
            BCF = Burst2Iaf(systeminfo, file_name.split('_length.csv')[0])
            BCF.get_cluster_result()
            new_class = dbscan_clu(BCF.action_list, eps=5)  # cluster
            agg_rule = rule_aggre(new_class)
            if not os.path.exists(os.path.join(fp_dir, 'idle_fp', systeminfo)):
                os.makedirs(os.path.join(fp_dir, 'idle_fp', systeminfo))

            with open(os.path.join(fp_dir, 'idle_fp', systeminfo, file_name.split('_length.csv')[0]), 'w') as f:
                a = json.dumps({num: list(rule_) for num, rule_ in enumerate(agg_rule)})
                json.dump(a, f)
