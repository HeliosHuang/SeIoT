import numpy as np
import os
from datetime import datetime
import time
import random
from Af_match import *

fp_dir = 'fpset'
random.seed(2022)


def read_trigger(mode, start_date, stop_date):
    trigger = []
    data_start = datetime.fromtimestamp(time.mktime(time.strptime(start_date, "%Y-%m-%d %H:%M:%S"))).timestamp()
    data_stop = datetime.fromtimestamp(time.mktime(time.strptime(stop_date, "%Y-%m-%d %H:%M:%S"))).timestamp()
    if mode == 'time':
        file_name = 'time_trigger_record.csv'
    elif mode == 'motion':
        file_name = 'motion_trigger_record.csv'
    elif mode == 'stop':
        file_name = 'motion_stop_record.csv'
    else:
        return 0
    with open(file_name, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            data_object = datetime.fromtimestamp(time.mktime(time.strptime(row[0].split('.')[0], "%Y-%m-%d %H:%M:%S")))
            if data_start <= data_object.timestamp() <= data_stop:
                trigger.append(data_object.timestamp())
    return trigger


def trigger_match(trigger, miss_d):
    passed_key = []
    unpassed_key = []
    for key in miss_d:
        total = 0
        for bur_time in miss_d[key]:
            for time_ in trigger:
                if abs(float(bur_time) - time_) <= 3:
                    total += 1
                    break
        if total/len(miss_d[key]) > 0.5 and len(miss_d[key]) > 0:
            passed_key.append(list(key))
        else:
            unpassed_key.append(list(key))
    return passed_key, unpassed_key


def cal_jacc_co(set_1, set_2):
    return len(set_1.intersection(set_2))/len(set_1.union(set_2))


def read_length_feature(feature_dir, dev_name, tk):
    traffic_burst = []
    feature_file = os.path.join(feature_dir, dev_name, tk + '_length.csv')
    with open(feature_file, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            traffic_burst.append(row)
    return traffic_burst


def read_time_feature(feature_dir, dev_name, tk):
    traffic_burst_time = []
    feature_file = os.path.join(feature_dir, dev_name, tk + '_time.csv')
    with open(feature_file, 'r') as f:
        f_csv = csv.reader(f)
        for row in f_csv:
            traffic_burst_time.append(row)
    return traffic_burst_time


def dict2rule(rule_dict):
    out_dict = {}
    for num, key_ in enumerate(rule_dict):
        out_dict[num] = key_
    return out_dict


def flow2activefp(systeminfo, trigger, fp_dir):
    burst_file = os.listdir(os.path.join('feature_idle', systeminfo))
    for file_name in burst_file:
        if not file_name.endswith('_length.csv'):
            continue
        feature_dir = 'feature_active'
        action = read_agg_rule(systeminfo, file_name.split('_length.csv')[0], rc='idle')

        action_dict = {}
        for num, rule in enumerate(action):
            action_dict[rule] = set(action[rule])
        idle_all_set = set()
        for rule_ in action:
            idle_all_set = idle_all_set.union(set(action[rule_]))

        traffic_bur = read_length_feature(feature_dir, systeminfo, file_name.split('_length.csv')[0])
        traffic_time = read_time_feature(feature_dir, systeminfo, file_name.split('_length.csv')[0])
        mis_num = 0
        mis_dic = {}
        for num, bur in enumerate(traffic_bur):
            bur = [int(i) for i in bur]
            bur = set(bur)
            res_idle = softMatch(action_dict, bur)
            if res_idle == []:
                mis_num += 1
                if tuple(bur) not in mis_dic:
                    mis_dic[tuple(bur)] = [traffic_time[num][0]]
                else:
                    mis_dic[tuple(bur)].append(traffic_time[num][0])

        time_trigger = read_trigger('time', '2022-01-01 00:00:00', '2022-01-02 23:59:00')
        motion_trigger = read_trigger('motion', '2022-01-01 00:00:00', '2022-01-02 23:59:00')
        stop_trigger = read_trigger('stop', '2022-01-01 00:00:00', '2022-01-02 23:59:00')

        if trigger == 'time':
            new_bur, up_key = trigger_match(time_trigger, mis_dic)
        else:
            new_bur, up_key = trigger_match(motion_trigger + stop_trigger, mis_dic)

        new_dic = {}
        kp_list = []
        for rule_ in new_bur:
            diff_set = set(rule_).difference(idle_all_set)
            if diff_set == set():
                continue
            for kp in diff_set:
                if tuple([kp]) not in new_dic:
                    kp_list.append(kp)
                    new_dic[tuple([kp])] = [set(rule_), set(rule_)]

        active_dict = dict2rule(new_dic)
        new_idle_bur = []
        for rule_ in up_key:
            for kp in kp_list:
                if kp not in rule_:
                    new_idle_bur.append(rule_)
        for idle_rule in new_idle_bur:
            jacc = -1
            for num, rule_ in enumerate(action_dict):
                if cal_jacc_co(set(idle_rule), set(action_dict[rule_])) > jacc:
                    jacc = cal_jacc_co(set(idle_rule), set(action_dict[rule_]))
                    rule_match = rule_
            action_dict[rule_match] = action_dict[rule_match].union(set(idle_rule))

        if not os.path.exists(os.path.join(fp_dir, 'active_fp', systeminfo)):
            os.makedirs(os.path.join(fp_dir, 'active_fp', systeminfo))
        with open(os.path.join(fp_dir, 'active_fp', systeminfo, file_name.split('_length.csv')[0]), 'w') as f:
            a = json.dumps({k: list(active_dict[k]) for k in active_dict})
            json.dump(a, f)
        with open(os.path.join(fp_dir, 'idle_fp', systeminfo, file_name.split('_length.csv')[0]), 'w') as f:
            a = json.dumps({k: list(action_dict[k]) for k in action_dict})
            json.dump(a, f)


if __name__ == "__main__":
    dev_name = {
        'fan': 'motion',
        'heater': 'time',
        'dehumidifier': 'time',
        'humidifier_der': 'time',
        'mijia_humidifier': 'time',
        'desklight': 'motion',
        'bedlight': 'motion',
        'ac_plug': 'time',
        'smartplug_1': 'motion',
    }
    for dev in dev_name:
        print(dev)
        flow2activefp(dev, dev_name[dev], fp_dir)