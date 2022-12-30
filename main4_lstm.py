import time
import csv

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils import *

# 选择网络训练使用的设备
if torch.cuda.is_available():
    print('PyTorch is using Cuda Device')
    device = torch.device('cuda')
else:
    print('PyTorch is using CPU Device')
    device = torch.device('cpu')

trainPaths = {
    r'train\201902.txt', r'train\202002.txt', r'train\202102.txt', r'train\202103.txt', 
    r'train\202104.txt', r'train\202105.txt', r'train\202106.txt', r'train\202107.txt', 
    r'train\202108.txt', r'train\202109.txt', r'train\202110.txt', r'train\202111.txt', 
    r'train\202112.txt'
}
testPaths = {r'test\test.txt'}

batch_size = 1024
hidden_dim = 256

def train_process1():
    # 加载训练集
    trainSet = DatasetLabel(trainPaths)
    train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
    # 加载测试集
    testSet = DatasetLabel(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = LSTMNetwork(8, hidden_dim, 1).to(device)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    # 训练网络
    epochs = 10
    best_Recall_P = 0
    # 保存日志
    with open('historys/m4_1_history.csv', 'a+', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'Epoch', 'Cost(min)', 'Recall(N)', 'Recall(P)', 'Threshold'])
    for epoch in range(epochs):
        begin_time = time.time()
        # 网络训练
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out = network.forward(batch_x.to(torch.float32).to(device), device)
            loss = loss_fun(out, batch_y.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 测试网络 能保留99%的有效数据
        y_P_hat, y_N_hat = [], []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                y_ = network.forward(batch_x.to(torch.float32).to(device), device)
                y = batch_y.detach().cpu().numpy()[0][0]
                # 收集预测结果，按原始标签分类
                if y > 0.5:
                    y_P_hat.append(y_.detach().cpu().numpy()[0][0])
                else:
                    y_N_hat.append(y_.detach().cpu().numpy()[0][0])
        Recall_N, Recall_P, threshold = cal_score(y_N_hat.copy(), y_P_hat.copy(), 0.99)
        # 输出信息
        end_time = time.time()
        used_time = (end_time - begin_time) / 60
        print(f'Epoch {epoch+1}/{epochs}: cost({used_time:.2f} min), Recall(N) {Recall_N:.3f}, Recall(P) {Recall_P:.3f}, Threshold {threshold:.3f}')
        # 保存模型
        if Recall_P > best_Recall_P:
            best_Recall_P = Recall_P
            network.save(f'weights/m4_1_weight.pt')
        # 保存日志
        with open('historys/m4_1_history.csv', 'a+', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch+1, used_time, Recall_N, Recall_P, threshold])

    print('训练完成!')

def train_process2():
    # 加载训练集
    trainSet = DatasetLabel2(trainPaths)
    train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
    # 加载测试集
    testSet = DatasetLabel2(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = LSTMNetwork(8, hidden_dim, 2).to(device)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    # 训练网络
    epochs = 10
    best_Recall_P = 0
    # 保存日志
    with open('historys/m4_2_history.csv', 'a+', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'Epoch', 'Cost(min)', 'Recall(N)', 'Recall(P)', 'Threshold'])
    for epoch in range(epochs):
        begin_time = time.time()
        # 网络训练
        for step, (batch_x, batch_y) in enumerate(train_loader):
            out = network.forward(batch_x.to(torch.float32).to(device), device)
            loss = loss_fun(out, batch_y.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 测试网络 能保留99%的有效数据
        y_N_hat, y_P_hat = [], []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                y_ = network.forward(batch_x.to(torch.float32).to(device), device)
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
            network.save(f'weights/m4_2_weight.pt')
        # 保存日志
        with open('historys/m4_2_history.csv', 'a+', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch+1, used_time, Recall_N, Recall_P, threshold])

    print('训练完成!')

def train_process3():
    # 加载训练集
    trainSet = DatasetResidue(trainPaths, reject=True)
    train_loader = DataLoader(dataset=trainSet, batch_size=batch_size, shuffle=True)
    # 加载测试集
    testSet = DatasetResidue(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = LSTMNetwork(7, hidden_dim, 1).to(device)
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    # 训练参数
    epochs = 10
    best_Recall_P = 0
    # 保存日志
    with open('historys/m4_3_history.csv', 'a+', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'Epoch', 'Cost(min)', 'MAE(N)', 'MSE(N)', 'Recall(N)', 'MAE(P)', 'MSE(P)', 'Recall(P)', 'Threshold'])
    for epoch in range(epochs):
        begin_time = time.time()
        # 训练网络
        for step, (batch_x, batch_y, batch_z) in enumerate(train_loader):
            out = network.forward(batch_x.to(torch.float32).to(device), device)
            loss = loss_fun(out, batch_y.float().to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # 测试网络
        y_P_hat, y_N_hat = [], []
        for step, (batch_x, batch_y, batch_z) in enumerate(test_loader):
            with torch.no_grad():
                y_ = network.forward(batch_x.to(torch.float32).to(device), device)
                if batch_z.detach().cpu().numpy()[0][0] < 0.5:
                    y_N_hat.append(abs(batch_y.detach().cpu().numpy()[0][0] - y_.detach().cpu().numpy()[0][0])*83)
                else:
                    y_P_hat.append(abs(batch_y.detach().cpu().numpy()[0][0] - y_.detach().cpu().numpy()[0][0])*83)
        MAEN = np.mean(y_N_hat)
        MSEN = np.mean(np.power(y_N_hat, 2))
        MAEP = np.mean(y_P_hat)
        MSEP = np.mean(np.power(y_P_hat, 2))
        Recall_N, Recall_P, threshold = cal_score(y_N_hat.copy(), y_P_hat.copy(), 0.99)
        # 输出信息
        end_time = time.time()
        used_time = (end_time - begin_time) / 60
        print(f'Epoch {epoch+1}/{epochs}: cost({used_time:.2f} min), Threshold({threshold:.3f})')
        print(f'Negative: MAE({MAEN:.3f}), MSE({MSEN:.3f}), Recall({Recall_N:.3f})')
        print(f'Positive: MAE({MAEP:.3f}), MSE({MSEP:.3f}), Recall({Recall_P:.3f})')
        # 保存网络        
        if Recall_P > best_Recall_P:
            best_Recall_P = Recall_P
            network.save(f'weights/m4_3_weight.pt')
        # 保存日志
        with open('historys/m4_3_history.csv', 'a+', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch+1, used_time, MAEN, MSEN, Recall_N, MAEP, MSEP, Recall_P, threshold])
    print('训练完成!')

if __name__ == '__main__':   
    # train_process1() 
    # train_process2()
    train_process3()

'''
训练样本: 9570734
测试样本: 50001
阴性样本标签 [0]
阳性样本标签 [1]
'''
