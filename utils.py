import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import math

def cal_score2(y, y_hat):
    TN = 0  # real = 0, hat = 0
    FN = 0  # real = 1, hat = 0
    FP = 0  # real = 0, hat = 1
    TP = 0  # real = 1, hat = 1

    for i in range(len(y)):
        if y[i] == 0 and y_hat[i] == 0:   # 质控正常，预测正常
            TN += 1
        if y[i] == 1 and y_hat[i] == 0:   # 质控异常，预测正常
            FN += 1
        if y[i] == 0 and y_hat[i] == 1:   # 质控正常，预测异常
            FP += 1
        if y[i] == 1 and y_hat[i] == 1:   # 质控异常，预测异常
            TP += 1
    return TN, FN, FP, TP

def cal_score(y_N_hat, y_P_hat, N_th_Recall, reverse=False):
    '''
        y_N_hat 标签为阴性的样本的预测值
        y_P_hat 标签为阳性的样本的预测值
        N_th_Recall 标签为阴性的样本的回召率
        return1 剔除异常值后, 标签为阴性的样本与剔除前的比例
        return2 剔除异常值后, 标签为阳性的样本与剔除前的比例
        return3 实际阈值 > 该值的都被剔除了
    '''
    y_N_num = len(y_N_hat)
    y_P_num = len(y_P_hat)
    del_data = []
    if reverse:
        # 样本从大到小排序
        y_N_hat.sort(reverse=reverse)
        y_P_hat.sort(reverse=reverse)  
        # 剔除数据
        while True:        
            if len(y_P_hat) == 0 or len(y_N_hat) == 0:
                break
            # 抽样最小的结果
            max_y_N_hat = y_N_hat[-1]
            max_y_P_hat = y_P_hat[-1]
            # 比较结果
            if max_y_P_hat < max_y_N_hat:
                # 阳性比阴性小，去除阳性结果
                # y_P_hat.pop()
                del_data.append(y_P_hat.pop())
            if max_y_P_hat > max_y_N_hat:
                # 阳性比阴性大，去除阴性结果
                if (len(y_N_hat)-1) / y_N_num < N_th_Recall:
                    # 保留99%的有效数据
                    break
                else:
                    # y_N_hat.pop()
                    del_data.append(y_N_hat.pop())
            if max_y_P_hat == max_y_N_hat:
                # 一样大
                if (len(y_N_hat)-1) / y_N_num < N_th_Recall:
                    # 保留99%的有效数据
                    break
                else:
                    # y_N_hat.pop()
                    # y_P_hat.pop()
                    del_data.append(y_N_hat.pop())
                    del_data.append(y_P_hat.pop())
        return len(y_N_hat)/y_N_num, (y_P_num - len(y_P_hat))/y_P_num, min(y_N_hat)
    else:
        # 样本从小到大排序
        y_N_hat.sort()
        y_P_hat.sort()
        # 剔除数据
        while True:        
            if len(y_P_hat) == 0 or len(y_N_hat) == 0:
                break
            # 抽样最大的结果
            max_y_N_hat = y_N_hat[-1]
            max_y_P_hat = y_P_hat[-1]
            # 比较结果
            if max_y_P_hat > max_y_N_hat:
                # 阳性比阴性大，去除阳性结果
                y_P_hat.pop()
            if max_y_P_hat < max_y_N_hat:
                # 阴性比阳性大，去除阴性结果
                if (len(y_N_hat)-1) / y_N_num < N_th_Recall:
                    # 保留99%的有效数据
                    break
                else:
                    y_N_hat.pop()
            if max_y_P_hat == max_y_N_hat:
                # 一样大
                if (len(y_N_hat)-1) / y_N_num < N_th_Recall:
                    # 保留99%的有效数据
                    break
                else:
                    y_N_hat.pop()
                    y_P_hat.pop()
        return len(y_N_hat)/y_N_num, (y_P_num - len(y_P_hat))/y_P_num, max(y_N_hat)

class DatasetLabel(Dataset):
    def __init__(self, paths):
        self.x = []
        self.y = []
        for path in paths:
            print(f'加载 {path} 中...')
            with open(path, 'r') as file:
                for line in file:
                    year, mon, day, hour, min, lat, long, sst, qcs  = self.split_data(line)
                    self.x.append(np.array([year, mon, day, hour, min, lat, long, sst]))
                    self.y.append(np.array([qcs]))
        print(f'样本数量: {len(self.x)}')

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

    def split_data(self, str):
        '''
            brief 分割并转换数据
            str 数据字符串
            return  year, mon, day, hour, min, lat, long, sst, qcs 
        '''
        # 按空格分割字符串
        words = str.split()
        # 数据提取
        year    = (float(words[0][0:4]) - 2019) / 2 # 年
        mon     = float(words[0][4:6]) / 12     # 月
        day     = float(words[0][6:8]) / 31     # 日
        hour    = float(words[0][8:10]) / 24    # 小时
        min     = float(words[0][10:12]) / 60   # 分钟
        lat     = (float(words[1]) + 90) / 180  # 纬度
        long    = float(words[2]) / 360         # 经度
        sst     = (float(words[3]) + 33) / 83   # 海表温度
        qcs     = 1.0 if len(words) > 4 else 0.0    # 质控符

        return year, mon, day, hour, min, lat, long, sst, qcs 

class DatasetLabel2(Dataset):
    def __init__(self, paths):
        self.x = []
        self.y = []
        for path in paths:
            print(f'加载 {path} 中...')
            with open(path, 'r') as file:
                for line in file:
                    year, mon, day, hour, min, lat, long, sst, qcs  = self.split_data(line)
                    self.x.append(np.array([year, mon, day, hour, min, lat, long, sst]))
                    self.y.append(np.array(qcs))
        print(f'样本数量: {len(self.x)}')

    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    def __len__(self):
        return len(self.y)

    def split_data(self, str):
        '''
            brief 分割并转换数据
            str 数据字符串
            return  year, mon, day, hour, min, lat, long, sst, qcs 
        '''
        # 按空格分割字符串
        words = str.split()
        # 数据提取
        year    = (float(words[0][0:4]) - 2019) / 2 # 年
        mon     = float(words[0][4:6]) / 12     # 月
        day     = float(words[0][6:8]) / 31     # 日
        hour    = float(words[0][8:10]) / 24    # 小时
        min     = float(words[0][10:12]) / 60   # 分钟
        lat     = (float(words[1]) + 90) / 180  # 纬度
        long    = float(words[2]) / 360         # 经度
        sst     = (float(words[3]) + 33) / 83   # 海表温度(-33 ~ 50)
        if len(words) > 4:                      # 质控符
            qcs = [0, 1]    # 阳性
        else:
            qcs = [1, 0]    # 阴性

        return year, mon, day, hour, min, lat, long, sst, qcs 

class DatasetResidue(Dataset):
    def __init__(self, paths, reject=False):
        self.x = []
        self.y = []
        self.z = []
        # 加载数据
        for path in paths:
            print(f'加载 {path} 中...', end='')
            valid = 0
            total = 0
            with open(path, 'r') as file:
                for line in file:
                    total += 1
                    year, mon, day, hour, min, lat, long, sst, qcs  = self.split_data(line)
                    if reject and qcs == 0:
                        self.x.append(np.array([year, mon, day, hour, min, lat, long]))
                        self.y.append(np.array([sst]))
                        self.z.append(np.array([qcs]))
                        valid += 1
                    elif not reject:
                        self.x.append(np.array([year, mon, day, hour, min, lat, long]))
                        self.y.append(np.array([sst]))
                        self.z.append(np.array([qcs]))
                        valid += 1
            print(f' {valid}/{total}')

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.z[index]
    
    def __len__(self):
        return len(self.y)

    def split_data(self, str):
        '''
            brief 分割并转换数据
            str     数据字符串
            return  year, mon, day, hour, min, lat, long, sst, qcs 
        '''
        # 按空格分割字符串
        words = str.split()
        # 数据提取
        year    = (float(words[0][0:4]) - 2019) / 2 # 年
        mon     = float(words[0][4:6]) / 12         # 月
        day     = float(words[0][6:8]) / 31         # 日
        hour    = float(words[0][8:10]) / 24        # 小时
        min     = float(words[0][10:12]) / 60       # 分钟
        lat     = (float(words[1]) + 90) / 180      # 纬度
        long    = float(words[2]) / 360             # 经度
        sst     = (float(words[3]) + 33) / 83       # 海表温度
        qcs     = 1 if len(words) > 4 else 0        # 质控符

        return year, mon, day, hour, min, lat, long, sst, qcs 

class LSTMNetwork(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int):
        super(LSTMNetwork, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data, device):
        h0 = torch.zeros(1, self.hidden_dim).to(device)
        c0 = torch.zeros(1, self.hidden_dim).to(device)

        out1, _ = self.lstm(data, (h0, c0))
        out2 = self.net(out1)
        return out2
    
    def save(self, path: str = ''):
        if len(path) > 0:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), 'temp.pt')

class NormalNetwork(nn.Module):
    def __init__(self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int):
        super(NormalNetwork, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, data):
        out = self.net(data)
        return out
    
    def save(self, path: str = ''):
        if len(path) > 0:
            torch.save(self.state_dict(), path)
        else:
            torch.save(self.state_dict(), 'temp.pt')
