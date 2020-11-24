import torch
import torch.nn as nn
from torch.optim import SGD
import math
import numpy as np
import json
import glob
import datetime
import csv
import os

DATASET_DIRPATH = "../video/result/"
# ハイパーパラメータ
hidden_size = 400
epochs_num = 40
batch_size = 1024
learning_rate = 0.01

class my_lstm(nn.Module):
    """
    NNのモデル
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(my_lstm, self).__init__()
        
        # LSTMの設定
        # batch_firstは入力するデータ系列の次元がbatch_sizeの頭にくることを指す
        self.rnn = nn.LSTM(input_size = input_dim, hidden_size = hidden_dim, batch_first = True) 
        # 全結合の設定
        self.linear = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, inputs, hidden0=None):
        output, (hidden, cell) = self.rnn(inputs, hidden0)# LSTM層
        output = self.linear(output) # 全結合層
        output = torch.sigmoid(output)
        return output

def get_landmarks(json_load, offset):
    landmarks = []
    for l in range(len(json_load[0])):
        landmarks.append(json_load[offset][l]["x"])
        landmarks.append(json_load[offset][l]["y"])
        landmarks.append(json_load[offset][l]["z"])
    return landmarks

def make_dataset(json_load, content):
    """
    データセットを作る
    params:
        dataset_path : データセットの相対パス
    """
    train_data = []
    train_label = []
    
    # ラベル
    label = 0.0
    if content == "wolf":
        label = 1.0
    # データ
    for offset in range(len(json_load)): # 5290
        # dataset.jsonを読み込む
        landmarks = []
        for l in range(len(json_load[0])): # 検出ごとのデータ 468
            #print(json_load[offset][l]["x"])
            landmarks.append(json_load[offset][l]["x"])
            landmarks.append(json_load[offset][l]["y"])
            landmarks.append(json_load[offset][l]["z"])
        train_data.append(landmarks)
        train_label.append(label)
        
    return train_data, train_label
    
def make_batch(train_data, train_label, batch_iter):
    """
    データセットをバッチ化させる
    """
    batch_data = []
    batch_label = []
    
    for idx in range(batch_iter, batch_iter+batch_size):
        batch_data.append(train_data[idx])
        batch_label.append(train_label[idx])
    
    return torch.tensor([batch_data]), torch.tensor([batch_label])
        
def read_json(json_path):
    path = os.path.join(DATASET_DIRPATH, json_path)
    with open(path, "r") as f:
        json_load = json.load(f)
        return json_load

# def train(model, train_data, train_label):
    

def test(model, test_data, test_label):
    test_size = len(test_data)
    test_accuracy = 0.0
    for i in range(int(test_size / batch_size)):
        offset = i * batch_size
        data, label = torch.tensor([test_data[offset:offset+batch_size]]), torch.tensor([test_label[offset:offset+batch_size]])
        
        output = model(data, None)
        output = torch.squeeze(output, 0)
        output = torch.squeeze(output, 1)
        label = torch.squeeze(label, 0)
        
        test_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.1)

    test_accuracy /= test_size
    return test_accuracy

def make_identificate_dataset(wolf_json_load, human_json_load):
    train_wolf_size = int(len(wolf_json_load) * 0.8)
    train_human_size = int(len(human_json_load) * 0.8)
    
    train_data = []
    train_label = []
    test_w_data = []
    test_w_label = []
    test_h_data = []
    test_h_label = []
    
    for offset in range(train_wolf_size):
        train_data.append(get_landmarks(wolf_json_load, offset))
        train_label.append(1.0)
        
    for offset in range(train_human_size):
        train_data.append(get_landmarks(human_json_load, offset))
        train_label.append(0.0)
        
    for offset in range(train_wolf_size, len(wolf_json_load)):
        test_w_data.append(get_landmarks(wolf_json_load, offset))
        test_w_label.append(1.0)
    
    for offset in range(train_human_size, len(human_json_load)):
        test_h_data.append(get_landmarks(human_json_load, offset))
        test_h_label.append(0.0)
        
    return train_data, train_label, test_w_data, test_w_label, test_h_data, test_h_label
    
def outlier_detection():
    """
    # 異常値検出による方法
    # データセット.jsonの読み込み
    train_json_load = read_json("naoki_1_1105_13_22_facemesh_output_dataset.json") # human 
    # train_json_load = read_json("naoki_3_1105_13_38_Trim2_output_dataset.json") # wolf 
    test1_json_load = read_json("naoki_3_1105_13_38_Trim_output_dataset.json") # wolf
    # test2_json_load = read_json("kazuma_2_1105_13_29_output_dataset.json") # human 
    
    train_data, train_label = make_dataset(train_json_load, "human")
    test1_data, test1_label = make_dataset(test1_json_load, "wolf")
    # test2_data, test2_label = make_dataset(test2_json_load, "human")
    """
    wolf_json_load = read_json("naoki_3_1105_13_38_Trim_output_dataset.json")
    human_json_load = read_json("naoki_1_1105_13_22_facemesh_output_dataset.json")

    train_data, train_label, test_w_data, test_w_label, test_h_data, test_h_label = make_identificate_dataset(wolf_json_load, human_json_load)
    
    input_size = len(train_data[0])
    training_size = len(train_data) - ( len(train_data) % batch_size) 
    
    model = my_lstm(input_size, hidden_size, 1)
    
    criterion = nn.BCELoss()
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    for epoch in range(epochs_num):
        # training
        running_loss = 0.0
        training_accuracy = 0.0
        batch_iter = 0
        for i in range(int(training_size / batch_size)):
            optimizer.zero_grad()
            
            data, label = make_batch(train_data, train_label, batch_iter)
            
            batch_iter += batch_size
            if batch_iter+batch_size >= training_size :
                batch_iter = 0
            
            output = model(data)
            
            loss = criterion(output, label)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # 次元を下げる
            output = torch.squeeze(output, 0)
            output = torch.squeeze(output, 1)
            label = torch.squeeze(label, 0)
            
            training_accuracy += np.sum(np.abs((output.data - label.data).numpy()) < 0.5)
            
        training_accuracy /= training_size
        
        # test1_accuracy = test(model, test1_data, test1_label)
        # test2_accuracy = test(model, test2_data, test2_label)
        # test3_accuracy = test(test3_data, test3_label)
        # test4_accuracy = test(test4_data, test4_label)
        test_w_accuracy = test(model, test_w_data, test_w_label)
        test_h_accuracy = test(model, test_h_data, test_h_label)
        
        # csvファイル
        dt_now = datetime.datetime.now()    
        result_path = "../train_result/result_" + dt_now.strftime("%Y_%m%d_%H_%M_%S") + ".csv"
        result_csv_open = open(result_path, "w")
        result_writer = csv.writer(result_csv_open)
        
        # print("epoch: %d loss: %.5f, training_accuracy: %.5f, test_accuracy: %.5f" % (epoch + 1, running_loss, training_accuracy, test1_accuracy))
        # result_writer.writerow([epoch+1, running_loss, training_accuracy, test1_accuracy])
        
        print("%d loss: %.5f, training_accuracy: %.5f, test_w_accuracy: %.5f, test_h_accuracy: %.5f" % (epoch + 1, running_loss, training_accuracy, test_w_accuracy, test_h_accuracy))
        result_writer.writerow([epoch+1, running_loss, training_accuracy, test_w_accuracy, test_h_accuracy])        
        
    result_csv_open.close()
    
def identifyer():
    # 識別機による方法
    # データセット.jsonの読み込み
    wolf_json_load = read_json("naoki_3_1105_13_38_Trim_output_dataset.json")
    human_json_load = read_json("kazuma_2_1105_13_29_output_dataset.json")

    train_data, train_label, test_w_data, test_w_label, test_h_data, test_h_label = make_identificate_dataset(wolf_json_load, human_json_load)

    
if __name__ == "__main__":
    # main()
    outlier_detection()
    # identifyer()