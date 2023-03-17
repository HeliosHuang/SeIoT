from Iaf_extractor import *
import numpy as np
from Aaf_extractor import *
import time
import sys

fp_dir = 'fpset'


def hardMatch(ruleset, burst):
    matchedRules = set()
    for rule in ruleset:
        if ruleset[rule].issubset(burst):
            if matchedRules == set():
                matchedRules = rule
            elif len(ruleset[rule]) > len(ruleset[matchedRules]):
                matchedRules = rule
    if matchedRules != set():
        return matchedRules
    else:
        return set()


def softMatch(ruleset, burst):
    matchedRules = []
    for rule in ruleset:
        if burst.issubset(ruleset[rule]):
            matchedRules = rule
            break
    if matchedRules == []:
        for rule_1 in ruleset:
            for rule_2 in ruleset:
                if rule_2 <= rule_1 or matchedRules != []:
                    continue
                if burst.issubset(ruleset[rule_1].union(ruleset[rule_2])):
                    matchedRules = (deepcopy(rule_1), deepcopy(rule_2))
    return matchedRules


def singleMatch(burst, activeRule, idleRule):
    matched_activeRules = hardMatch(activeRule, burst)
    if matched_activeRules != set():
        flag = 0
        for element_ in burst.difference([list(activeRule[key_])[0] for key_ in activeRule]):
            if softMatch(idleRule, {element_}) == []:
                flag = 1
        if flag == 0:
            return [matched_activeRules, [], set()]
        else:
            return [set(), [], set()]
    matched_idleRules = softMatch(idleRule, burst)
    return [set(), matched_idleRules, set()]


def read_agg_rule(systeminfo, tk, rc='all'):

    if rc == 'idle':
        with open(os.path.join(fp_dir, 'idle_fp', systeminfo, tk), 'r') as fingerprint:
            a = json.load(fingerprint)
            a = json.loads(a)
            idle_rule = {int(k): set(a[k]) for k in a}
        return idle_rule
    with open(os.path.join(fp_dir, 'active_fp', systeminfo, tk), 'r') as fingerprint:
        a = json.load(fingerprint)
        a = json.loads(a)
        active_rule = {int(k): set(a[k]) for k in a}
    with open(os.path.join(fp_dir, 'idle_fp', systeminfo, tk), 'r') as fingerprint:
        a = json.load(fingerprint)
        a = json.loads(a)
        idle_rule = {int(k): set(a[k]) for k in a}
    return active_rule, idle_rule


def burstmatch(systeminfo, mode='idle_test'):
    if mode == 'idle_test':
        feature_dir = 'feature_idle_test'
    if mode == 'active_test':
        feature_dir = 'active_burst'
    if mode == 'obtain_thre':
        feature_dir = 'feature_active'
    burst_file = os.listdir(os.path.join(feature_dir, systeminfo))
    time_list = []
    fp_num = 0
    all_num = 0
    active_num = 0
    idle_num = 0
    for file_name in burst_file:
        if not file_name.endswith('_length.csv'):
            continue

        active_rule, idle_rule = read_agg_rule(systeminfo, file_name.split('_length.csv')[0])
        traffic_bur = read_length_feature(feature_dir, systeminfo, file_name.split('_length.csv')[0])
        if mode != 'active_test':
            traffic_time = read_time_feature(feature_dir, systeminfo, file_name.split('_length.csv')[0])
        max_rule = 0
        for rule_ in idle_rule:
            if len(idle_rule[rule_]) > max_rule:
                max_rule = len(idle_rule[rule_])
        idle_match_res = {k: 0 for k in idle_rule}

        idle_final_res = {k: [] for k in idle_rule}
        false_positive = []
        former_time = -1
        for num, bur in enumerate(traffic_bur):
            all_num += 1
            if mode == 'obtain_thre':
                if former_time == -1:
                    former_time = float(traffic_time[num][0])
            bur_len = len(bur)
            bur = [int(i) for i in bur]
            bur = set(bur)
            time_start = time.time()
            match_res = singleMatch(bur, active_rule, idle_rule)
            time_end = time.time()

            time_list.append(time_end-time_start)
            if match_res[0] != set():
                active_num += 1
            if isinstance(match_res[1], int):
                idle_match_res[match_res[1]] += bur_len
                idle_num += 1
            elif isinstance(match_res[1], tuple):
                idle_match_res[match_res[1][0]] += bur_len
                idle_match_res[match_res[1][1]] += bur_len
                idle_num += 1
            if match_res == [set(), [], set()]:
                fp_num += 1
                false_positive.append(bur)
            if mode == 'obtain_thre':
                if float(traffic_time[num][0]) - former_time > 300:
                    for key_ in idle_match_res:
                        idle_final_res[key_].append(idle_match_res[key_])
                    idle_match_res = {k: 0 for k in idle_rule}
                    former_time = float(traffic_time[num][0])

        i = 0
        thre_all = []
        if mode == 'obtain_thre':
            for key_ in idle_final_res:
                trigger_num = list(np.array(idle_final_res[key_])[np.nonzero(idle_final_res[key_])])
                trigger_num.sort()
                if len(trigger_num) > 0:
                    i+=1
                    thre_ = trigger_num[-1]
                else:
                    thre_ = 0
                thre_all.append(deepcopy(thre_))
            print(file_name.split('_length.csv')[0])
            print(thre_all)
    if mode != 'obtain_thre':
        print('false alarm rate:')
        print(fp_num/all_num)
        print('active burst number:')
        print(active_num)
        print('idle burst number:')
        print(idle_num)


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
    random.seed(2022)
    if sys.argv[1] == 't':
        for dev_ in dev_name:
            print('xxxxxxxxxxxxxxxxxxx')
            print(dev_)
            burstmatch(dev_, mode='obtain_thre')
    if sys.argv[1] == 'a':
        for dev_ in dev_name:
            print('xxxxxxxxxxxxxxxxxxx')
            print(dev_)
            burstmatch(dev_, mode='active_test')
    if sys.argv[1] == 'i':
        for dev_ in dev_name:
            print('xxxxxxxxxxxxxxxxxxx')
            print(dev_)
            burstmatch(dev_, mode='idle_test')


