import glob
import json

def check_dataset():
    dataset_path = "../video/result/naoki_1_1105_13_22_facemesh_output_dataset.json"
    
    json_open = open(dataset_path, "r")
    json_load = json.load(json_open)
    
    print("データの数: " + str(len(json_load)))
    print("１回のデータの数: " + str(len(json_load[0])))
    
if __name__=="__main__":
    check_dataset()