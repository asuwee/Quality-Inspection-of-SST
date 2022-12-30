import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

def train_process():
    trainPaths = {
        r'train\201902.txt', r'train\202002.txt', r'train\202102.txt', r'train\202103.txt', 
        r'train\202104.txt', r'train\202105.txt', r'train\202106.txt', r'train\202107.txt', 
        r'train\202108.txt', r'train\202109.txt', r'train\202110.txt', r'train\202111.txt', 
        r'train\202112.txt'
    }
    testPaths = {r'test\test.txt'}

    # 选择网络训练使用的设备
    if torch.cuda.is_available():
        print('PyTorch is using Cuda Device')
        device = torch.device('cuda')
    else:
        print('PyTorch is using CPU Device')
        device = torch.device('cpu')
    # 加载训练集
    trainSet = DatasetLabel2(trainPaths)
    train_loader = DataLoader(dataset=trainSet, batch_size=1024, shuffle=True)
    # 加载测试集
    testSet = DatasetLabel2(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = NormalNetwork(8, 256, 2).to(device)
    # network.load_state_dict(torch.load('weights/m2_weight.pt'))
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    # 训练网络
    epochs = 10
    best_Recall_P = 0
    # 保存日志
    with open('historys/m2_history.csv', 'a+', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'Epoch', 'Recall(N)', 'Recall(P)', 'Threshold'])
    for epoch in range(epochs):
        begin_time = time.time()
        # 网络训练
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out = network.forward(batch_x.to(torch.float32).to(device))
            loss = loss_fun(out, batch_y.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 测试网络 能保留99%的有效数据
        y_N_hat, y_P_hat = [], []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                y_ = network.forward(batch_x.to(torch.float32).to(device))
                batch_y = batch_y.detach().cpu().numpy()[0]
                # 收集预测结果，按原始标签分类
                if batch_y[0] > batch_y[1]:
                    # Negative  [1, 0]
                    y_N_hat.append(y_.detach().cpu().numpy()[0])
                else:
                    # Postive   [0, 1]
                    y_P_hat.append(y_.detach().cpu().numpy()[0])

        # 点距离直线x=y的距离 D=(x-y)/sqrt(2)
        y_N_hat_2xy = []
        y_P_hat_2xy = []
        for index in range(len(y_N_hat)):
            dis_2_N = math.sqrt(math.pow(1 - y_N_hat[index][0], 2) + math.pow(0 - y_N_hat[index][1], 2)) # 越大，阳性概率越大
            dis_2_P = math.sqrt(math.pow(0 - y_N_hat[index][0], 2) + math.pow(1 - y_N_hat[index][1], 2)) # 越大，阴性概率越大
            y_N_hat_2xy.append((dis_2_P-dis_2_N)/math.sqrt(2))
        for index in range(len(y_P_hat)):
            dis_2_N = math.sqrt(math.pow(1 - y_P_hat[index][0], 2) + math.pow(0 - y_P_hat[index][1], 2)) # 越大，阳性概率越大
            dis_2_P = math.sqrt(math.pow(0 - y_P_hat[index][0], 2) + math.pow(1 - y_P_hat[index][1], 2)) # 越大，阴性概率越大
            y_P_hat_2xy.append((dis_2_P-dis_2_N)/math.sqrt(2))
        Recall_N, Recall_P, threshold = cal_score(y_N_hat_2xy.copy(), y_P_hat_2xy.copy(), 0.99, reverse=True)
        # 输出信息
        end_time = time.time()
        used_time = (end_time - begin_time) / 60
        print(f'Epoch {epoch+1}/{epochs}: cost({used_time:.2f} min), Recall(N) {Recall_N:.3f}, Recall(P) {Recall_P:.3f}, Threshold {threshold:.3f}')
        # 保存模型
        if Recall_P > best_Recall_P:
            best_Recall_P = Recall_P
            network.save(f'weights/m2_weight.pt')
        # 保存日志
        with open('historys/m2_history.csv', 'a+', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch+1, Recall_N, Recall_P, threshold])

    print('训练完成!')

if __name__ == '__main__':   
    train_process() 
