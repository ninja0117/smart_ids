# 1d convolution
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import csv
import numpy as np
import matplotlib.pyplot as plt
import os

# 设置迭代次数
Epoch = 10
batch_size = 10
np.random.seed(4396)
acc_test = 0

# KDDTrain+_20Percent 25192 / 25192
# KDDTest-21 8110 / 11850
# KDDTrain+ 125973 / 125973
# KDDTest+ 18794 / 22544
#train_num = np.random.permutation(np.arange(494021))
train_num = np.random.permutation(np.arange(141832))  # 训练数据集中训练数据的数量
train_num_tensor = torch.tensor(train_num)
test_num = np.random.permutation(np.arange(72263))  # 测试集中测试数据的数量
test_num_tensor = torch.tensor(test_num)

def load_data(root):
    feature = []
    label = []
    print('----------------------------------')
    # 用于测试集构建
    time_start = time.time()
    file_path = root
    with open(file_path, 'r') as data_from:
        csv_reader = csv.reader(data_from)
        for item in csv_reader:
            # print i
            tmp = list(map(float, item[:25]))
            feature.append(tmp)
            label.append(int(item[25]))  # 添加标签（以独热向量表示）
    time_end = time.time()
    delta = time_end-time_start
    print('数据读取总用时: %fs'%(delta))
    return feature, label

def data_assign(feature, label):
    print('---------------------------------------')
    print('准备进行对读取数据的预处理：')
    feature_test = torch.tensor(feature, dtype=torch.float32)
    label_test = torch.tensor(label, dtype=torch.int64)
    return feature_test, label_test

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        # class torch.nn.Conv1d(in_channels, out_channels, kernel_size,
        # stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.conv4 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.lstm1 = nn.LSTM(input_size=11, hidden_size=48, num_layers=2, batch_first=True)
        # self.conv3 = nn.Conv1d(in_channels=128, out_channels=256, kernel_size=3)
        self.maxpool = nn.MaxPool1d(kernel_size=2, ceil_mode=True)
        # 1 * 256 * x
        self.fc1 = nn.Linear(6144, 1024)
        self.fc2 = nn.Linear(1024, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)
        out = F.gelu(out) # 1, 64, 39
#        out = self.maxpool(out)# 1, 64, 20
        out = self.conv2(out)
        out = F.gelu(out)
        out = self.maxpool(out)
        # out = self.conv3(out)
        # out = F.gelu(out)
        # out = self.conv4(out)
        # out = F.gelu(out)
        # out = self.maxpool(out)
        out, (h, c) = self.lstm1(out) # 1, 64, 70
        # Flatten()
        out = out.view(in_size, -1)

        out = F.dropout(out, p=0.25)
        out = self.fc1(out)
        out = F.gelu(out)
        out = F.dropout(out, p=0.5)
        out = self.fc2(out)
        out = F.gelu(out)
        out = self.softmax(out)
        return out

def train(model, feature_train, label_train, optimizer, criterion, epoch, Device):
#def train(model, train_iter, optimizer, epoch, Device):
    acc = []
    model.train()
    count = 0
    correct = 0
    loss_sum = 0
#    correct1 = 0
#    for id, (data, target) in enumerate(zip(feature_train, label_train)):
    for id, idx in enumerate(train_num):  # 乱序训练
        data, target = feature_train[idx], label_train[idx]
#    for id, (data, target) in enumerate(train_iter):
        count += 1
        data, target = data.to(Device), target.to(Device)
        target = target.unsqueeze(dim=0)
        data = data.unsqueeze(dim=0)
        data = data.unsqueeze(dim=0)
        optimizer.zero_grad()
        output = model(data)  # torch.Size([1, 23])
#        output = output.squeeze()
#        print(output.shape)
#        print(output)
        loss = criterion(output, target)
        loss_sum += loss
        loss.backward()
        optimizer.step()
        output1 = int(F.log_softmax(output, dim=1).max(1, keepdim=True)[1])  # torch.Size([1, 1]) 返回的是最大值的索引位置
        correct += (1 if target == output1 else 0)

#        correct += (1 if target.argmax(dim=1) == output.argmax(dim=1) else 0)
        if (id+1) % 100 == 0:
            print('Train Epoch: {},Loss: {:.6f}, Accuracy: {} / {} ({:.4f}%)'.format(
                epoch, loss_sum.item()/100. , correct, count, 100. * correct / count))
            acc.append(100. * correct / count)
            loss_sum = 0
    plt.plot(acc, c = 'red', label='pred')
    plt.ylabel('acc')
    plt.xlabel('number')
    plt.rcParams['figure.dpi'] = 300 # 设置图片分辨率为 1800*1200
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(r'./pic\pic%s.png'%epoch)
    plt.show()

def test(model, feature_train, label_train, criterion, epoch, Device):
    global acc_test
    loss = 0
    model.eval()
    count = 0
    correct = 0
    #    for id, (data, target) in enumerate(zip(feature_train, label_train)):
    for id, idx in enumerate(test_num):  # 乱序训练
        data, target = feature_train[idx], label_train[idx]
        count += 1
        data, target = data.to(Device), target.to(Device)
        target = target.unsqueeze(dim=0)
        data = data.unsqueeze(dim=0)
        data = data.unsqueeze(dim=0)
        output = model(data)  # torch.Size([1, 23])
        loss += criterion(output, target).item()
        output = int(F.log_softmax(output, dim=1).max(1, keepdim=True)[1])  # torch.Size([1, 1])
        correct += (1 if target == output else 0)
        #correct += (1 if target.argmax(dim=1) == output.argmax(dim=1) else 0)
    print('Train Epoch: {},Loss: {:.6f}, Accuracy: {} / {} ({:.4f}%)'.format(
        epoch, loss/count, correct, count, 100. * correct / count))
    tmp_acc = 100. * correct / count
    if acc_test < tmp_acc:
        try:
             state = {
                'net': model.state_dict(),
               'epoch': epoch,
               'name': 'LSTM'
            }

             if not os.path.isdir(r'./model'):
                os.mkdir(r'./model')
             torch.save(state, r'./model/dl-ids.pth')
        except Exception as e:
          print("发生了错误：", e)
        else:
            print("保存成功！")
        acc_test = tmp_acc

    return 100. * correct / count


if __name__ == '__main__':
    # KDDTrain+_20Percent
    # KDDTest-21

    # KDDTrain+
    # KDDTest+
    print("准备进行训练集读取：")
    feature_train, label_train = load_data(root = r'./UNSW_NB15_training-set-v4.csv')
    print("准备进行测试集读取：")
    feature_test, label_test = load_data(root = r'./UNSW_NB15_testing-set-v4.csv')
    # 对训练集和测试集进行拷贝
    feature_train, label_train = data_assign(feature_train, label_train)
    feature_test, label_test = data_assign(feature_test, label_test)
    print(feature_train.shape)

    # -----------------------
#    train_iter = torch.utils.data.DataLoader((feature_train, label_train), batch_size=batch_size,
#                                               shuffle=True, drop_last=True, num_workers=4)
#    test_iter = torch.utils.data.DataLoader((feature_test, label_test), batch_size=batch_size,
#                                               shuffle=True, drop_last=True, num_workers=4)
    #------------------------

    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ConvNet().to(Device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    print('---------------------------------------')
    print('准备进行训练和测试：')
    test_acc = []
    for epoch in range(1, Epoch+1):
        time_start = time.time()
        train(model, feature_train, label_train, optimizer, criterion, epoch, Device)
#        train(model, train_iter, optimizer, epoch, Device)
        time_end = time.time()
        test_acc.append(test(model, feature_test, label_test, criterion, epoch, Device))
        delta = time_end - time_start
        print('训练一轮用时: %fs' % (delta))
        print('第{}轮迭代完成'.format(epoch))
    print("全部训练结束！")
    plt.plot(test_acc, c = 'red', label='pred')
    plt.ylabel('acc')
    plt.xlabel('number')
    plt.rcParams['figure.dpi'] = 300 # 设置图片分辨率为 1800*1200
    plt.rcParams['savefig.dpi'] = 300
    plt.savefig(r'./pic/pic-acc.png')
    plt.show()