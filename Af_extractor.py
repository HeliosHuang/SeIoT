from Iaf_extractor import *
from Aaf_extractor import *

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

fp_dir = 'fpset'
for systeminfo in dev_name:
    print(systeminfo)
    # extract idle afs
    burst_file = os.listdir(os.path.join('feature_idle', systeminfo))
    for file_name in burst_file:
        if not file_name.endswith('_length.csv'):
            continue
        feature_dir = 'feature_idle'
        BCF = Burst2Iaf(systeminfo, file_name.split('_length.csv')[0])
        BCF.get_cluster_result()
        new_class = dbscan_clu(BCF.action_list)
        agg_rule = rule_aggre(new_class)
        if not os.path.exists(os.path.join(fp_dir, 'idle_fp', systeminfo)):
            os.makedirs(os.path.join(fp_dir, 'idle_fp', systeminfo))
        with open(os.path.join(fp_dir, 'idle_fp', systeminfo, file_name.split('_length.csv')[0]), 'w') as f:
            a = json.dumps({num: list(rule_) for num, rule_ in enumerate(agg_rule)})
            json.dump(a, f)
    # extract active afs
    flow2activefp(systeminfo, dev_name[systeminfo], fp_dir)
