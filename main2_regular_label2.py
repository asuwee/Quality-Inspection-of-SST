import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import csv
from utils import *

def train_process():
    '''
        使用全连接神经网络区分正常和异常数据
    '''
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
    loss_fun = nn.MSELoss()
    optimizer = torch.optim.Adam(network.parameters(), lr=0.0003)
    # 训练网络
    epochs = 10
    best_acc_P = 0
    # 保存日志
    with open('historys/m2_history.csv', 'a+', encoding='utf-8', newline='') as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow([f'Epoch', 'Acc(N)', 'Acc(P)'])
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
        y_N_num, y_P_num = 0, 0
        y_N_hat, y_P_hat = [], []
        for step, (batch_x, batch_y) in enumerate(test_loader):
            with torch.no_grad():
                y_ = network.forward(batch_x.to(torch.float32).to(device))
                batch_y = batch_y.detach().cpu().numpy()[0]
                y_ = y_.detach().cpu().numpy()[0]
                # 1. 统计原始样本中，阳性和阴性数据的数量
                # 2. 收集预测结果，按原始标签分类
                if batch_y[0] > batch_y[1]:
                    # Negative  [1, 0]
                    y_N_num += 1
                    y_N_hat.append(y_[1] - y_[0])
                else:
                    # Postive   [0, 1]
                    y_P_num += 1
                    y_P_hat.append(y_[1] - y_[0])
        acc_N, acc_P, threshold = cal_score(y_P_num, y_N_num, y_P_hat, y_N_hat, 0.99)
        # 输出信息
        end_time = time.time()
        used_time = (end_time - begin_time) / 60
        print(f'Epoch {epoch+1}/{epochs}: cost({used_time:.2f} min), Recall(N) {acc_N:.3f}, Recall(P) {acc_P:.5f}, Threshold {threshold:.5f}')
        # 保存模型
        if acc_P > best_acc_P:
            best_acc_P = acc_P
            network.save(f'weights/m2_weight.pt')
        # 保存日志
        with open('historys/m2_history.csv', 'a+', encoding='utf-8', newline='') as file:
            csv_writer = csv.writer(file)
            csv_writer.writerow([epoch+1, acc_N, acc_P])

    print('训练完成!')

if __name__ == '__main__':   
    train_process() 

'''
训练样本: 9570734
测试样本: 50001
阴性样本标签 [0]
阳性样本标签 [1]
'''
