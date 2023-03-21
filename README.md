# SeIoT
This repository contains the implementation of the main modules of SeIoT. The .pcap dataset and full codes will be published in an open access manner after acceptance.

## Testbed
The floorplan of our testbed:
![contents](floorplan.JPG)

## Code architecture
Core module code, including AF module and Anomaly detection module, not including packet parser module and feature extraction module. Instead, we provide the obtained .csv files generated by them.
```
Action Fingerprint (AF) module:
    -- AF extractor: AF_extractor.py
        -- idle AF extractor: Iaf_extractor.py
        -- active AF extractor: Aaf_extractor.py
        
    -- AF match: AF_match.py
    
Anomaly Detection module:
    -- Time-related anomaly detection: AF_match.py
    
    -- Interaction-related anomaly detection
        -- device state anomaly detection: node_c.py
            -- GAT layer: model_GAT.py
                -- graph attention layer: layers.py
        -- environment state anomaly detection: env_c.py
            -- GAT layer: model_GAT.py
                -- graph attention layer: layers_for_pre.py
```
## Dataset architecture
Note that the dataset only include .csv files, because .pcap files involve user privacy and may violate double blindness principle.
```
Packet length and timestamp sequences:
    -- feature_idle: traffic bursts in 7 days, only include idle traffic
        -- device names: select only one for the same device
            -- Trace key: (dev_r, port_d, port_r, protocol)
            
    -- feature_idle_test: traffic bursts in 21 days, only include idle traffic
        -- device names: select only one for the same device
            -- Trace key: (dev_r, port_d, port_r, protocol)
            
    -- feature_active: traffic bursts in 2 days, include both idle traffic and active traffic
        -- device names: select only one for the same device
            -- Trace key: (dev_r, port_d, port_r, protocol)
            
    -- active_burst: manually marked active bursts in 2 days, only include active traffic
        -- device names: select only one for the same device
            -- Trace key: (dev_r, port_d, port_r, protocol)
            
Trigger_record: corresponding to feature_active, used for Correlation Inspection
    -- Time_trigger_record.csv
    -- Motion_trigger_record.csv
    -- Motion_stop_record.csv
    
State_record: home states in 14 days, 7 days for training and 7 days for testing
    -- environment information extracted from OCR results
        -- temperature, humidity, motion, light
    -- device state extracted from traffic via AF
```
## Instructions

### Af_extractor
Af_extractor.py is a script to extract both idle AFs and active AFs. You can use this script to generate AFs as follows. The generated AFs will be stored in '/fpset'.
```sh
python AF_extractor.py
```

### Af_match
You can use Af_match.py to obtain thre<sub>AF</sub>, which will be used for time-related anomaly detection:
```sh
python AF_match.py t
```
You can use AF_match.py to test the recall and precision of idle and active traffic:
```sh
python AF_match.py a
python AF_match.py i
```

### node_c
You can use node_c.py to train and test device state anomaly detection. The model trained in our evaluation stored in 'pre_trained_model/node_c_model'
```sh
python node_c.py t
python AF_match.py te
```

### env_c
You can use env_c.py to train and test environment state anomaly detection. The model trained in our evaluation stored in 'pre_trained_model/env_c_model'
```sh
python env_c.py t
python env_c.py te
```

