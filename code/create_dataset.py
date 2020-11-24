import os
import sys
import glob
import json

argv = sys.argv

# 第一引数に作成したい動画の名前（MediaPipeで出力したフォルダ名）
video_name = argv[0]

MEDIAPIPE_OUTPUT_DIRPATH = "../video/result/"
FACE_MESH_LANDMARKS_DIRPAHT = "/face_mesh/landmarks/"
POSE_LANDMARKS_DIRPATH = "/pose/landmarks/"

face_mesh_landmarks_dirpath = MEDIAPIPE_OUTPUT_DIRPATH + video_name + FACE_MESH_LANDMARKS_DIRPAHT
pose_landmarks_dirpath = MEDIAPIPE_OUTPUT_DIRPATH + video_name + POSE_LANDMARKS_DIRPATH

def create_dataset():
    print("face_mesh_landmarks_dirpath: " + face_mesh_landmarks_dirpath)
    face_mesh_flist = glob.glob(face_mesh_landmarks_dirpath + "*.json")
    dataset_json_fpath = MEDIAPIPE_OUTPUT_DIRPATH + video_name +"_dataset.json"
    print("dataset_json_path: " + dataset_json_fpath)
    
    dataset = []
    
    for f in face_mesh_flist:
        json_open = open(f, "r")
        json_load = json.load(json_open)
        
        #print(json_load["landmark"])
        
        data_t = []
        
        for j in json_load["landmark"]:
            pos = {"x": j["x"], "y": j["y"], "z": j["z"]}
            data_t.append(pos)
        
        dataset.append(data_t)
        
    with open(dataset_json_fpath, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    create_dataset()
