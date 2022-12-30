import csv
import matplotlib.pyplot as plt

def figure1():
    m1_history = []
    m2_history = []
    m3_history = []
    with open('historys/m1_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m1_history.append(float(row[2]))
            skipHeader = False

    with open('historys/m2_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m2_history.append(float(row[2]))
            skipHeader = False

    with open('historys/m3_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m3_history.append(float(row[6]))
            skipHeader = False

    maxRecall1 = max(m1_history)
    maxRecall1_epoch = m1_history.index(maxRecall1) + 1
    maxRecall2 = max(m2_history)
    maxRecall2_epoch = m2_history.index(maxRecall2) + 1
    maxRecall3 = max(m3_history)
    maxRecall3_epoch = m3_history.index(maxRecall3) + 1

    plt.plot(range(1, len(m1_history)+1), m1_history, label = 'Method 1 Only One Label')
    plt.text(maxRecall1_epoch, maxRecall1, f'{maxRecall1:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
    plt.plot(range(1, len(m2_history)+1), m2_history, label = 'Method 2 Two Label')
    plt.text(maxRecall2_epoch, maxRecall2, f'{maxRecall2:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
    plt.plot(range(1, len(m3_history)+1), m3_history, label = 'Method 3 Residue')
    plt.text(maxRecall3_epoch, maxRecall3, f'{maxRecall3:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))

    plt.xlim((1, 10))
    plt.ylim((0.5, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Recall(P)')
    plt.grid()
    plt.legend()
    plt.show()

def figure2():
    m1_history = []
    m2_history = []
    m3_history = []
    with open('historys/m4_1_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m1_history.append(float(row[3]))
            skipHeader = False

    with open('historys/m4_2_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m2_history.append(float(row[2]))
            skipHeader = False

    with open('historys/m4_3_history.csv', encoding='utf-8') as file:
        skipHeader = True
        for row in csv.reader(file, skipinitialspace=True):
            if not skipHeader:
                m3_history.append(float(row[7]))
            skipHeader = False

    maxRecall1 = max(m1_history)
    maxRecall1_epoch = m1_history.index(maxRecall1) + 1
    maxRecall2 = max(m2_history)
    maxRecall2_epoch = m2_history.index(maxRecall2) + 1
    maxRecall3 = max(m3_history)
    maxRecall3_epoch = m3_history.index(maxRecall3) + 1

    plt.plot(range(1, len(m1_history)+1), m1_history, label = 'Method 1 Only One Label (LSTM)')
    plt.text(maxRecall1_epoch, maxRecall1, f'{maxRecall1:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
    plt.plot(range(1, len(m2_history)+1), m2_history, label = 'Method 2 Two Label (LSTM)')
    plt.text(maxRecall2_epoch, maxRecall2, f'{maxRecall2:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
    plt.plot(range(1, len(m3_history)+1), m3_history, label = 'Method 3 Residue (LSTM)')
    plt.text(maxRecall3_epoch, maxRecall3, f'{maxRecall3:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))

    plt.xlim((1, 10))
    plt.ylim((0.0, 1))
    plt.xlabel('Epochs')
    plt.ylabel('Recall(P)')
    plt.grid()
    plt.legend()
    plt.show()

def figure3():
    import math
    import torch
    from torch.utils.data import DataLoader
    from utils import DatasetLabel2, NormalNetwork, cal_score
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
    # 加载测试集
    testSet = DatasetLabel2(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = NormalNetwork(8, 256, 2).to(device)
    network.load_state_dict(torch.load('weights/m2_weight.pt', map_location=device))
    # 测试网络 能保留99%的有效数据
    y_N_hat, y_P_hat = [], []
    m1, m2, m3, m4 = [], [], [], []
    for step, (batch_x, batch_y) in enumerate(test_loader):
        with torch.no_grad():
            y_ = network.forward(batch_x.to(torch.float32).to(device))
            batch_y = batch_y.detach().cpu().numpy()[0]
            # 收集预测结果，按原始标签分类
            if batch_y[0] > batch_y[1]:
                # Negative  [1, 0]
                y_N_hat.append(y_.detach().cpu().numpy()[0])
                m1.append(y_N_hat[-1][0])
                m2.append(y_N_hat[-1][1])
            else:
                # Postive   [0, 1]
                y_P_hat.append(y_.detach().cpu().numpy()[0])
                m3.append(y_P_hat[-1][0])
                m4.append(y_P_hat[-1][1])
    import matplotlib.pyplot as plt
    plt.subplot(1, 2, 1)
    plt.scatter(m1, m2, label='Negative Sample', color='#1f77b4')
    plt.xlabel('Negative')
    plt.ylabel('Positive')
    plt.xlim([-0.93, 1.36])
    plt.ylim([-0.32, 1.95])
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.scatter(m3, m4, label='Positive Sample', color='#ff7f0e')
    plt.xlabel('Negative')
    plt.ylabel('Positive')
    plt.xlim([-0.93, 1.36])
    plt.ylim([-0.32, 1.95])
    plt.grid()
    plt.legend()
    plt.show()

def figure4():
    import math
    import torch
    from torch.utils.data import DataLoader
    from utils import DatasetLabel2, NormalNetwork, cal_score
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
    # 加载测试集
    testSet = DatasetLabel2(testPaths)
    test_loader = DataLoader(dataset=testSet, batch_size=1, shuffle=False)
    # 创建网络
    network = NormalNetwork(8, 256, 2).to(device)
    network.load_state_dict(torch.load('weights/m2_weight.pt', map_location=device))
    # 测试网络 能保留99%的有效数据
    y_N_hat, y_P_hat = [], []
    m1, m2, m3, m4 = [], [], [], []
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
    m1, m2, m3, m4 = [], [], [], []
    for index in range(len(y_N_hat)):
        dis_2_N = math.sqrt(math.pow(1 - y_N_hat[index][0], 2) + math.pow(0 - y_N_hat[index][1], 2)) # 越大，阳性概率越大
        dis_2_P = math.sqrt(math.pow(0 - y_N_hat[index][0], 2) + math.pow(1 - y_N_hat[index][1], 2)) # 越大，阴性概率越大
        m1.append(dis_2_N)
        m2.append(dis_2_P)
        y_N_hat_2xy.append((dis_2_P-dis_2_N)/math.sqrt(2))
    for index in range(len(y_P_hat)):
        dis_2_N = math.sqrt(math.pow(1 - y_P_hat[index][0], 2) + math.pow(0 - y_P_hat[index][1], 2)) # 越大，阳性概率越大
        dis_2_P = math.sqrt(math.pow(0 - y_P_hat[index][0], 2) + math.pow(1 - y_P_hat[index][1], 2)) # 越大，阴性概率越大
        m3.append(dis_2_N)
        m4.append(dis_2_P)
        y_P_hat_2xy.append((dis_2_P-dis_2_N)/math.sqrt(2))
    plt.subplot(1, 2, 1)
    plt.plot([math.sqrt(2) * 0.564, math.sqrt(2) * 0.564 + 10], [0, 10], color='black')
    plt.scatter(m2, m1, label='Negative Sample', color='#1f77b4')
    plt.xlabel('Negative: Distance To Posittve')
    plt.ylabel('Posittve: Distance To Negative')
    plt.xlim([0, 1.8])
    plt.ylim([0, 2.7])
    plt.grid()
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot([math.sqrt(2) * 0.564, math.sqrt(2) * 0.564 + 10], [0, 10], color='black')
    plt.scatter(m4, m3, label='Positive Sample', color='#ff7f0e')
    plt.xlabel('Negative: Distance To Posittve')
    plt.ylabel('Posittve: Distance To Negative')
    plt.xlim([0, 1.8])
    plt.ylim([0, 2.7])
    plt.grid()
    plt.legend()
    plt.show()

figure1()
figure2()
figure3()
figure4()