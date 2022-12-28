import csv
import matplotlib.pyplot as plt

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
plt.text(maxRecall1_epoch-0.4, maxRecall1+0.02, f'{maxRecall1:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
plt.plot(range(1, len(m2_history)+1), m2_history, label = 'Method 2 Two Label')
plt.text(maxRecall2_epoch-0.4, maxRecall2+0.02, f'{maxRecall2:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))
plt.plot(range(1, len(m3_history)+1), m3_history, label = 'Method 3 Residue')
plt.text(maxRecall3_epoch-0.4, maxRecall3+0.02, f'{maxRecall3:.3f}', bbox=dict(facecolor='white', alpha=1, boxstyle='round'))

plt.xlim((1, 10))
plt.ylim((0.5, 1))
plt.xlabel('Epochs')
plt.ylabel('Recall(P)')
plt.grid()
plt.legend()
plt.show()