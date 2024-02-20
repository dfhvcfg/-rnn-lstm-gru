import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from  torch.utils.data import Dataset, DataLoader
import string
import time
import matplotlib.pyplot as plt


# 获取常用字符包括字母和常用标点  # zhang
all_letters = string.ascii_letters + " .,;'"
# 获取常用字符数量
n_letters = len(all_letters)
print("字母表 n_letter--->", all_letters, n_letters)


# 国家名 种类数
categorys = ['Italian', 'English', 'Arabic', 'Spanish', 'Scottish', 'Irish', 'Chinese', 'Vietnamese', 'Japanese',
             'French', 'Greek', 'Dutch', 'Korean', 'Polish', 'Portuguese', 'Russian', 'Czech', 'German']
# 国家名 个数
categorynum = len(categorys)

print('国家名数目categorys', len(categorys), categorys)


# 思路分析
# 1 打开数据文件 open(filename, mode='r', encoding='utf-8')
# 2 按行读文件、提取样本x 样本y line.strip().split('\t')
# 3 返回样本x的列表、样本y的列表 my_list_x, my_list_y
def read_data(filename):

    my_list_x, my_list_y = [], []

    # 1 打开数据文件 open(filename, mode='r', encoding='utf-8')
    with open(filename, mode='r', encoding='utf-8') as f :
        # print(f.readlines())
        # 2 按行读文件、提取样本x 样本y line.strip().split('\t')
        for line in f.readlines():
            (x, y) = line.strip().split('\t')
            my_list_x.append(x)
            my_list_y.append(y)

    # 3 返回样本x的列表、样本y的列表 my_list_x, my_list_y
    return  my_list_x, my_list_y


# 原始数据 -> 数据源NameClassDataset --> 数据迭代器DataLoader
# 构造数据源 NameClassDataset，把语料转换成x y
# 1 init函数 设置样本x和y self.my_list_x self.my_list_y 条目数self.sample_len
# 2 __len__(self)函数  获取样本条数
# 3 __getitem__(self, index)函数 获取第几条样本数据
    # 按索引 获取数据样本 x y
    # 样本x onehot张量化 torch.zeros(len(x), n_letters)
    # 遍历人名 的 每个字母 做成one-hot编码 tensor_x[li][all_letters.find(letter)] = 1
    # 样本y 张量化 torch.tensor(categorys.index(y), dtype=torch.long)
    # 返回tensor_x, tensor_y
class NameClassDataset(Dataset):

    def __init__(self, my_list_x, my_list_y):
        self.my_list_x = my_list_x
        self.my_list_y = my_list_y
        # 数据源的条目数
        self.sample_len = len(my_list_y)

    # 获取数据源长度
    def __len__(self):
        return self.sample_len

    # 根据index获取第几条数据
    def __getitem__(self, index):

        # 按索引 获取数据样本 x y
        x = self.my_list_x[index] # 'zhang'
        y = self.my_list_y[index]

        # 样本x onehot张量化 torch.zeros(len(x), n_letters)
        tensor_x = torch.zeros(len(x), n_letters) # [5, 57]

        # 遍历人名 的 每个字母 做成one-hot编码 tensor_x[li][all_letters.find(letter)] = 1
        for li, letter in enumerate(x):
            tensor_x[li][all_letters.find(letter)] = 1

        # 样本y 张量化 torch.tensor(categorys.index(y), dtype=torch.long)
        tensor_y = torch.tensor(categorys.index(y), dtype=torch.long )

        # 返回tensor_x, tensor_y
        return tensor_x, tensor_y


def dm01_test_NameClassDataset():

    # 1 读数据到内存
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')
    # print(len(my_list_x), len(my_list_y), my_list_x[0:3], my_list_y[0:3])

    # 2 实例化数据源
    mynameclassdataset = NameClassDataset(my_list_x, my_list_y)
    print('mynameclassdataset-->', mynameclassdataset)  # cmd+shift+u

    # 3 实例化dataloader
    mydataloader = DataLoader(dataset=mynameclassdataset, batch_size=1, shuffle=True)
    print('mydataloader-->', mydataloader)

    # 4 遍历数据
    for x, y in mydataloader:
        print('x-->', x.shape, x)
        print('y-->', y.shape, y)


# RNN类 实现思路分析：
# 1 init函数 (self, input_size, hidden_size, output_size, num_layers=1):
    # 准备super() self.rnn self.linear self.softmax=nn.LogSoftmax(dim=-1)
# 2 forward(input, hidden)函数
    # 数据调形 [6,57] ---> [6,1,57] input.unsqueeze(1)
    # 数据经过rnn层 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
        # rr, hn=self.rnn(input, hidden)
    # 取最后一个128当人名的特征 [6,1,128] --> [1,128]  tmprr = rr[-1]
    # 数据经过全连接层 [1,128] ---> [1,18]
    # 返回 self.softmax(tmprr), hn
# 3 初始化隐藏层输入数据 inithidden()
#   形状[self.num_layers, 1, self.hidden_size]
class RNN(nn.Module):
    #                   57          128             18
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化rnn层
        self.rnn = nn.RNN(self.input_size, self.hidden_size, self.num_layers)

        # 实例化线性层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):

        # 数据调形 [6,57] ---> [6,1,57] input.unsqueeze(1)
        input = input.unsqueeze(1)

        # 数据经过rnn层 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
        rr, hn = self.rnn(input, hidden)

        # 取最后一个128当人名的特征 [6,1,128] --> [1,128]  tmprr = rr[-1]
        tmprr = rr[-1]

        # 数据经过全连接层 [1,128] ---> [1,18]
        tmprr = self.linear(tmprr)

        # 返回 self.softmax(tmprr), hn
        return self.softmax(tmprr), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


def dm02_test_RNN():
    # 1 实例化模型
    myrnn = RNN(57, 128, 18)
    print('myrnn-->', myrnn)

    # 2 准备数据 input hidden
    input = torch.randn(6, 57)
    hidden = myrnn.inithidden()

    # 3 给模型喂数据
    # 3-1 一次性的送数据
    output, hn = myrnn(input, hidden)
    print('一次性的送数据output-->', output.shape, output)
    # print('hn-->', hn.shape, hn)

    # 3-2 一个字符一个字符的给模型送数据
    print('一个字符一个字符的给模型送数据')
    hidden = myrnn.inithidden()

    for i in range(input.shape[0]):
        tmpinput = input[i] # 1维数据
        # print('tmpinput-->', tmpinput.shape) # 数据形状是[57,]

        # tmpinput.unsqueeze(0) [57,] --> [1, 57]
        output, hidden = myrnn(tmpinput.unsqueeze(0), hidden)
        print('output-->', output.shape, output)


### LSTM
class LSTM(nn.Module):
    #                   57          128             18
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化rnn层
        self.lstm = nn.LSTM(self.input_size, self.hidden_size, self.num_layers)

        # 实例化线性层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden, c0):

        # 数据调形 [6,57] ---> [6,1,57] input.unsqueeze(1)
        input = input.unsqueeze(1)

        # 数据经过rnn层 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
        rr, (hn, cn) = self.lstm(input, (hidden, c0))

        # 取最后一个128当人名的特征 [6,1,128] --> [1,128]  tmprr = rr[-1]
        tmprr = rr[-1]

        # 数据经过全连接层 [1,128] ---> [1,18]
        tmprr = self.linear(tmprr)

        # 返回 self.softmax(tmprr), hn
        return self.softmax(tmprr), hn, cn

    def inithidden(self):
        c0 = h0 = torch.zeros(self.num_layers, 1, self.hidden_size)
        return h0, c0


def dm03_test_LSTM():
    # 1 实例化模型
    mylstm = LSTM(57, 128, 18)
    print('mylstm-->', mylstm)

    # 2 准备数据 input hidden
    input = torch.randn(6, 57)
    hidden, c0 = mylstm.inithidden()

    # 3 给模型喂数据
    # 3-1 一次性的送数据
    output, hn, cn = mylstm(input, hidden, c0)
    print('一次性的送数据output-->', output.shape, output)
    # print('hn-->', hn.shape, hn)

    # 3-2 一个字符一个字符的给模型送数据
    print('一个字符一个字符的给模型送数据')
    hidden, c = mylstm.inithidden()

    for i in range(input.shape[0]):
        tmpinput = input[i] # 1维数据
        # print('tmpinput-->', tmpinput.shape) # 数据形状是[57,]

        # tmpinput.unsqueeze(0) [57,] --> [1, 57]
        output, hidden, c = mylstm(tmpinput.unsqueeze(0), hidden, c)
        print('output-->', output.shape, output)


### gru
class GRU(nn.Module):
    #                   57          128             18
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        # 实例化rnn层
        self.gru = nn.GRU(self.input_size, self.hidden_size, self.num_layers)

        # 实例化线性层
        self.linear = nn.Linear(self.hidden_size, self.output_size)

        # softmax层
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, input, hidden):

        # 数据调形 [6,57] ---> [6,1,57] input.unsqueeze(1)
        input = input.unsqueeze(1)

        # 数据经过rnn层 形状变化 [seqlen,1,57],[1,1,128]) -> [seqlen,1,128],[1,1,128]
        rr, hn = self.gru(input, hidden)

        # 取最后一个128当人名的特征 [6,1,128] --> [1,128]  tmprr = rr[-1]
        tmprr = rr[-1]

        # 数据经过全连接层 [1,128] ---> [1,18]
        tmprr = self.linear(tmprr)

        # 返回 self.softmax(tmprr), hn
        return self.softmax(tmprr), hn

    def inithidden(self):
        return torch.zeros(self.num_layers, 1, self.hidden_size)


def dm04_test_GRU():
    # 1 实例化模型
    mygru = GRU(57, 128, 18)
    print('mygru-->', mygru)

    # 2 准备数据 input hidden
    input = torch.randn(6, 57)
    hidden = mygru.inithidden()

    # 3 给模型喂数据
    # 3-1 一次性的送数据
    output, hn = mygru(input, hidden)
    print('一次性的送数据output-->', output.shape, output)
    # print('hn-->', hn.shape, hn)

    # 3-2 一个字符一个字符的给模型送数据
    print('一个字符一个字符的给模型送数据')
    hidden = mygru.inithidden()

    for i in range(input.shape[0]):
        tmpinput = input[i] # 1维数据
        # print('tmpinput-->', tmpinput.shape) # 数据形状是[57,]

        # tmpinput.unsqueeze(0) [57,] --> [1, 57]
        output, hidden = mygru(tmpinput.unsqueeze(0), hidden)
        print('output-->', output.shape, output)


# 思路分析
# 从文件获取数据、实例化数据源对象nameclassdataset 数据迭代器对象mydataloader
# 实例化模型对象my_rnn 损失函数对象mycrossentropyloss=nn.NLLLoss() 优化器对象myadam
# 定义模型训练的参数
#       starttime total_iter_num total_loss  total_loss_list total_acc_num  total_acc_list
# 外层for循环 控制轮数 for epoch_idx in range(epochs)
# 内层for循环 控制迭代次数 for i, (x, y) in enumerate(mydataloader)
    # 给模型喂数据   # 计算损失  # 梯度清零 # 反向传播  # 梯度更新
    # 计算辅助信息
    # 累加总已训练样本 总损失 总准确数
    # 每100次求1下平均损失tmploss 存入total_loss_list列表中 方便画图
    # 每100次求1下平均准确数目tmpacc 存入total_acc_num列表中 方便画图
    # 每2000次训练 打印日志
    # print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx+1, tmploss, time.time()-starttime, tmpacc))
    # 其他 # 预测对错 i_predit_tag = (1 if torch.argmax(output_y).item() == y.item() else 0)
# 模型保存
    # torch.save(my_rnn.state_dict(), './my_rnn_model_%d.bin' % (epoch_idx + 1))
# 返回 平均损失列表total_loss_list, 时间total_time, 平均准确total_acc_list


# 模型训练参数
mylr = 1e-3
epochs = 1

def my_train_rnn():

    # 从文件获取数据、实例化数据源对象nameclassdataset 数据迭代器对象mydataloader
    # 1-1 读数据到内存
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 1-2 实例化数据源
    mynameclassdataset = NameClassDataset(my_list_x, my_list_y)

    # 1-3 实例化dataloader
    mydataloader = DataLoader(dataset=mynameclassdataset, batch_size=1, shuffle=True)

    # 实例化模型对象my_rnn 损失函数对象 mycrossentropyloss=nn.NLLLoss() 优化器对象myadam
    # 2-1 实例化模型
    my_rnn = RNN(57, 128, 18)
    print('my_rnn-->', my_rnn)

    # 2-2 损失函数对象
    mycrossentropyloss = nn.NLLLoss()

    # 2-3 优化器对象myadam
    myadam = optim.Adam(my_rnn.parameters(), lr = mylr)

    # 定义模型训练的参数
    starttime = time.time()     # 模型训练开始时间
    total_iter_num = 0          # 已经训练了多少样本
    total_loss  = 0             # 损失累加器
    total_loss_list = []        # 每100个样本把损失函数 list
    total_acc_num  = 0          # 已训练样本正确的个数
    total_acc_list = []         # 每100个样本 acc

    # 外层for循环 控制轮数 for epoch_idx in range(epochs)
    for epoch_idx in range(epochs):

        # 内层for循环 控制迭代次数 for i, (x, y) in enumerate(mydataloader)
        for i, (x, y) in enumerate(mydataloader):

            # 给模型喂数据
            output_y, hidden = my_rnn(x[0], my_rnn.inithidden())
            # 计算损失
            my_loss = mycrossentropyloss(output_y, y)
            # 梯度清零
            myadam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            myadam.step()

            # 计算辅助信息
            total_iter_num += 1

            # 累加总已训练样本 总损失 总准确数
            total_loss += my_loss
            itag = (1 if torch.argmax(output_y).item() == y.item() else 0)
            total_acc_num += itag

            if total_iter_num % 100 == 0:
                # 每100次求1下平均损失tmploss 存入total_loss_list列表中 方便画图
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)

                # 每100次求1下平均准确数目tmpacc 存入total_acc_num列表中 方便画图
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)

            if total_iter_num % 2000 == 0:
                # 每2000次训练 打印日志
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx+1, tmploss, time.time()-starttime, tmpacc))

        # 保存模型
        torch.save(my_rnn.state_dict(), './my_rnn_model_%d.bin' % (epoch_idx + 1))

    # 训练完毕
    # 返回 平均损失列表total_loss_list, 时间 total_time, 平均准确total_acc_list
    total_time = int(time.time() - starttime)
    return total_loss_list, total_time, total_acc_list


### lstm
def my_train_lstm():

    # 从文件获取数据、实例化数据源对象nameclassdataset 数据迭代器对象mydataloader
    # 1-1 读数据到内存
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 1-2 实例化数据源
    mynameclassdataset = NameClassDataset(my_list_x, my_list_y)

    # 1-3 实例化dataloader
    mydataloader = DataLoader(dataset=mynameclassdataset, batch_size=1, shuffle=True)

    # 实例化模型对象my_rnn 损失函数对象 mycrossentropyloss=nn.NLLLoss() 优化器对象myadam
    # 2-1 实例化模型
    my_lstm = LSTM(57, 128, 18)
    print('my_lstm-->', my_lstm)

    # 2-2 损失函数对象
    mycrossentropyloss = nn.NLLLoss()

    # 2-3 优化器对象myadam
    myadam = optim.Adam(my_lstm.parameters(), lr = mylr)

    # 定义模型训练的参数
    starttime = time.time()     # 模型训练开始时间
    total_iter_num = 0          # 已经训练了多少样本
    total_loss  = 0             # 损失累加器
    total_loss_list = []        # 每100个样本把损失函数 list
    total_acc_num  = 0          # 已训练样本正确的个数
    total_acc_list = []         # 每100个样本 acc

    # 外层for循环 控制轮数 for epoch_idx in range(epochs)
    for epoch_idx in range(epochs):

        # 内层for循环 控制迭代次数 for i, (x, y) in enumerate(mydataloader)
        for i, (x, y) in enumerate(mydataloader):

            # 给模型喂数据
            h, c = my_lstm.inithidden()
            output_y, h, c = my_lstm(x[0], h, c)
            # 计算损失
            my_loss = mycrossentropyloss(output_y, y)
            # 梯度清零
            myadam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            myadam.step()

            # 计算辅助信息
            total_iter_num += 1

            # 累加总已训练样本 总损失 总准确数
            total_loss += my_loss
            itag = (1 if torch.argmax(output_y).item() == y.item() else 0)
            total_acc_num += itag

            if total_iter_num % 100 == 0:
                # 每100次求1下平均损失tmploss 存入total_loss_list列表中 方便画图
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)

                # 每100次求1下平均准确数目tmpacc 存入total_acc_num列表中 方便画图
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)

            if total_iter_num % 2000 == 0:
                # 每2000次训练 打印日志
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx+1, tmploss, time.time()-starttime, tmpacc))

        # 保存模型
        torch.save(my_lstm.state_dict(), './my_lstm_model_%d.bin' % (epoch_idx + 1))

    # 训练完毕
    # 返回 平均损失列表total_loss_list, 时间 total_time, 平均准确total_acc_list
    total_time = int(time.time() - starttime)
    return total_loss_list, total_time, total_acc_list


### gru train
def my_train_gru():

    # 从文件获取数据、实例化数据源对象nameclassdataset 数据迭代器对象mydataloader
    # 1-1 读数据到内存
    my_list_x, my_list_y = read_data('./data/name_classfication.txt')

    # 1-2 实例化数据源
    mynameclassdataset = NameClassDataset(my_list_x, my_list_y)

    # 1-3 实例化dataloader
    mydataloader = DataLoader(dataset=mynameclassdataset, batch_size=1, shuffle=True)

    # 实例化模型对象my_rnn 损失函数对象 mycrossentropyloss=nn.NLLLoss() 优化器对象myadam
    # 2-1 实例化模型
    my_gru = GRU(57, 128, 18)
    print('my_gru-->', my_gru)

    # 2-2 损失函数对象
    mycrossentropyloss = nn.NLLLoss()

    # 2-3 优化器对象myadam
    myadam = optim.Adam(my_gru.parameters(), lr = mylr)

    # 定义模型训练的参数
    starttime = time.time()     # 模型训练开始时间
    total_iter_num = 0          # 已经训练了多少样本
    total_loss  = 0             # 损失累加器
    total_loss_list = []        # 每100个样本把损失函数 list
    total_acc_num  = 0          # 已训练样本正确的个数
    total_acc_list = []         # 每100个样本 acc

    # 外层for循环 控制轮数 for epoch_idx in range(epochs)
    for epoch_idx in range(epochs):

        # 内层for循环 控制迭代次数 for i, (x, y) in enumerate(mydataloader)
        for i, (x, y) in enumerate(mydataloader):

            # 给模型喂数据
            output_y, hidden = my_gru(x[0], my_gru.inithidden())
            # 计算损失
            my_loss = mycrossentropyloss(output_y, y)
            # 梯度清零
            myadam.zero_grad()
            # 反向传播
            my_loss.backward()
            # 梯度更新
            myadam.step()

            # 计算辅助信息
            total_iter_num += 1

            # 累加总已训练样本 总损失 总准确数
            total_loss += my_loss
            itag = (1 if torch.argmax(output_y).item() == y.item() else 0)
            total_acc_num += itag

            if total_iter_num % 100 == 0:
                # 每100次求1下平均损失tmploss 存入total_loss_list列表中 方便画图
                tmploss = total_loss / total_iter_num
                total_loss_list.append(tmploss)

                # 每100次求1下平均准确数目tmpacc 存入total_acc_num列表中 方便画图
                tmpacc = total_acc_num / total_iter_num
                total_acc_list.append(tmpacc)

            if total_iter_num % 2000 == 0:
                # 每2000次训练 打印日志
                print('轮次:%d, 损失:%.6f, 时间:%d，准确率:%.3f' % (epoch_idx+1, tmploss, time.time()-starttime, tmpacc))

        # 保存模型
        torch.save(my_gru.state_dict(), './my_gru_model_%d.bin' % (epoch_idx + 1))

    # 训练完毕
    # 返回 平均损失列表total_loss_list, 时间 total_time, 平均准确total_acc_list
    total_time = int(time.time() - starttime)
    return total_loss_list, total_time, total_acc_list


def dm05_test_train_rnn_lstm_gru():
    total_loss_list_rnn, total_time_rnn, total_acc_list_rnn = my_train_rnn()

    total_loss_list_lstm, total_time_lstm, total_acc_list_lstm = my_train_lstm()

    total_loss_list_gru, total_time_gru, total_acc_list_gru = my_train_gru()

    # 绘制损失对比曲线
    # 创建画布0
    plt.figure(0)
    # # 绘制损失对比曲线
    plt.plot(total_loss_list_rnn, label="RNN")
    plt.plot(total_loss_list_lstm, color="red", label="LSTM")
    plt.plot(total_loss_list_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_loss2.png')
    plt.show()

    # 绘制柱状图
    # 创建画布1
    plt.figure(1)
    x_data = ["RNN", "LSTM", "GRU"]
    y_data = [total_time_rnn, total_time_lstm, total_time_gru]
    # 绘制训练耗时对比柱状图
    plt.bar(range(len(x_data)), y_data, tick_label=x_data)
    plt.savefig('./img/RNN_LSTM_GRU_period2.png')
    plt.show()

    # 绘制准确率对比曲线
    plt.figure(2)
    plt.plot(total_acc_list_rnn, label="RNN")
    plt.plot(total_acc_list_lstm, color="red", label="LSTM")
    plt.plot(total_acc_list_gru, color="orange", label="GRU")
    plt.legend(loc='upper left')
    plt.savefig('./img/RNN_LSTM_GRU_acc2.png')
    plt.show()


if __name__ == '__main__':

    # dm01_test_NameClassDataset()
    # dm02_test_RNN()
    # dm03_test_LSTM()
    # dm04_test_GRU()
    # my_train_rnn()
    # my_train_lstm()
    # my_train_gru()
    dm05_test_train_rnn_lstm_gru()
    print('人名分类器 End')