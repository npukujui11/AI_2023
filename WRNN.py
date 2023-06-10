import numpy as np
import os

###############
###############
# Data Preprocessing #
###############
###############

# 获取指定路径 './input' 下的所有文件名，并将其存储在 txt_filenames 列表中
data = ''
txt_filenames = os.listdir(r'./input')

for filename in txt_filenames:  # 使用 for 循环遍历 txt_filenames 列表中的每个文件名
    txt_file = open('./input/' + filename, 'r', encoding='utf-8')  # 读取 txt_file 中的所有文本，并将结果赋值给变量 buf
    buf = txt_file.read()  # 读取 txt_file 中的所有文本，并将结果赋值给变量 buf
    data = data + "\n" + buf  # 将 buf 中的文本添加到 data 中
    txt_file.close()  # 关闭 txt_file 文件

chars = list(set(data))  # 输出 data 的数据类型
data_size, vocab_size = len(data), len(chars)  # 输出 chars的长度
print('data has %d characters, %d unique.' % (data_size, vocab_size))  # 输出 data 的长度和 chars 的长度

char_to_ix = {ch: i for i, ch in enumerate(chars)}  # 将 chars 中的字符转换为索引 index，并将结果存储在字典 char_to_ix 中
ix_to_char = {i: ch for i, ch in enumerate(chars)}  # 将 chars 中的索引 index 转换为字符，并将结果存储在字典 ix_to_char 中

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Model Initializing #
###############
###############

# 模型超参数（要修改的话，请修改这里 by Gu Rui）
hidden_size = 160  # Hidden layer size
seq_length = 30  # RNN sequence length
learning_rate = 1e-2  # Learning rate

"""
### 定义超参数
隐藏层是循环神经网络（RNN）中的一部分，决定了网络的容量和表示能力。
较大的隐藏层可以捕捉更复杂的模式，但也会增加计算开销和过拟合的风险。在实际应用中，隐藏层的大小一般在 50 到 1000 之间。

序列长度是循环神经网络（RNN）中的另一个重要参数。序列长度决定了每次训练和预测时网络能够看到多少个字符。
序列长度越长，网络就能够看到更多的上下文信息，但也会增加计算开销和训练时间。在实际应用中，序列长度一般在 25 到 100 之间。

学习率是控制网络权重更新速度的参数，决定了网络的学习速度。较大的学习率可以加快网络的学习速度，但也会增加训练过程中的不稳定性。
在实际应用中，学习率一般在 1e-2 到 1e-5 之间。
"""

# 初始化权重矩阵和bias
Ux = np.random.randn(hidden_size, vocab_size) * 0.01  # 输入到隐藏层a的权重
Wx = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层a到隐藏层a的权重
Vx = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层a到隐藏层b的权重
Rx = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层a到隐藏层b的权重
Tx = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层b到隐藏层b的权重
Qx = np.random.randn(vocab_size, hidden_size) * 0.01  # 隐藏层b到输出的权重

s1 = np.zeros((hidden_size, 1))  # 隐藏层a的偏置
s2 = np.zeros((hidden_size, 1))  # 隐藏层b的偏置
s3 = np.zeros((vocab_size, 1))  # 输出层的偏置

"""
-------------------------------------------------------------------------------
"""


###############
###############
# Loss Function #
###############
###############

def sigmoid(x):
    x = np.clip(x, -500, 500)  # 防止溢出
    return 1 / (1 + np.exp(-x))  # sigmoid函数


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 防止溢出
    return e_x / e_x.sum(axis=0)  # softmax函数


def lossFun(inputs, targets):
    """
  input: `inputs`,`targets`为整数列，对应模型的输入与输出结果.
  output: return `loss`, `dUx`, `dWx`, `dVx`, `dRx`, `dTx`, `dQx`, `ds1`, `ds2`, `ds3`为损失、模型参数的梯度。
  by Gu Rui

  这里实现一个WRNN（一种改进的RNN模型）的损失函数，该模型包含两个隐藏层，分别为a和b，其中a层为RNN，b层为LSTM。
  其结构见附图W-RNN作业.pdf, 其中：x为输入，a为RNN隐藏层，b为LSTM隐藏层，o为输出层，Ux、Wx、Vx、Rx、Tx、Qx为权重矩阵，s1、s2、s3为偏置。
  f1和f2为激活函数，分别为sigmoid和sigmoid（此处未定义f1和f2）f3为softmax激活函数。

  前向传播公式有（latex）：$a^t=f_1(Ux^t+Wa^{t-1}+s_1)$、$b^t=f_2(Vx^t+Ra^{t-1}+Tb^{t-1}+s_2)$、$o^t=f_3(Qx^t+Tb^t+s_3)$

  该函数实现了WRNN参数的初始化、前向传播、损失计算、反向传播。
  """
    x, a, b, o = {}, {}, {}, {}  # 初始化输入、隐藏层a、隐藏层b、输出层的字典
    a[-1] = np.zeros((hidden_size, 1))  # 初始化隐藏层a的初始状态
    b[-1] = np.zeros((hidden_size, 1))  # 初始化隐藏层b的初始状态
    loss = 0  # 初始化损失

    # 前向传播过程
    for t in range(len(inputs)):  # 对序列中的每个字符
        x = np.zeros((vocab_size, 1))  # 初始化输入
        x[inputs[t]] = 1  # 将当前字符对应的输入置为1
        a[t] = sigmoid(np.dot(Ux, x) + np.dot(Wx, a[t - 1]) + s1)  # 计算隐藏层a
        b[t] = sigmoid(np.dot(Vx, a[t]) + np.dot(Rx, a[t - 1]) + np.dot(Tx, b[t - 1]) + s2)  # 计算隐藏层b
        o[t] = softmax(np.dot(Qx, b[t]) + s3)  # 计算输出层
        loss += -np.log(o[t][targets[t], 0])  # softmax (cross-entropy loss) 计算损失

    # 后向传播过程
    dUx, dWx, dVx, dRx, dTx, dQx = np.zeros_like(Ux), np.zeros_like(Wx), np.zeros_like(Vx), \
        np.zeros_like(Rx), np.zeros_like(Tx), np.zeros_like(Qx)  # 初始化参数梯度

    ds1, ds2, ds3 = np.zeros_like(s1), np.zeros_like(s2), np.zeros_like(s3)  # 初始化偏置梯度

    for t in reversed(range(len(inputs))):  # 对序列中的每个字符
        do = o[t].copy()
        do[targets[t]] -= 1  # backprop into o

        dQx += np.dot(do, b[t].T)  # backprop into Qx
        ds3 += do  # backprop into s3

        dbt = np.dot(Qx.T, do)  # backprop into b
        dbt_raw = dbt * (1 - b[t] * b[t])  # backprop through tanh nonlinearity

        dVx += np.dot(dbt_raw, a[t].T)  # backprop into Vx
        dRx += np.dot(dbt_raw, a[t - 1].T)  # backprop into Rx
        dTx += np.dot(dbt_raw, b[t - 1].T)  # backprop into Tx
        ds2 += dbt_raw  # backprop into s2

        dat = np.dot(Vx.T, dbt_raw) + np.dot(Rx.T, dbt_raw)  # backprop into a
        dat_raw = dat * (1 - a[t] * a[t])  # backprop through tanh nonlinearity

        dUx += np.dot(dat_raw, x.T)  # backprop into Ux
        dWx += np.dot(dat_raw, a[t - 1].T)  # backprop into Wx
        ds1 += dat_raw  # backprop into s1

    for dparam in [dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3]:  # clip to mitigate exploding gradients
        np.clip(dparam, -5, 5, out=dparam)  # clip to mitigate exploding gradients

    return loss, dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3  # 返回损失和参数梯度


"""
-------------------------------------------------------------------------------
"""


###############
###############
# Result Sampling #
###############
###############

# 定义模型的采样参数
def sample(hprev, seed_ix, n):
    """
  input: 从模型中抽取一个整数序列，长度为`n`，`hprev`是前一个时间步的隐藏状态，`seed_ix`是第一个时间步的种子字母。
  output: 返回一个整数序列
  by Gu Rui
  """
    x = np.zeros((vocab_size, 1))  # 初始化输入
    x[seed_ix] = 1  # 将第一个时间步的输入置为1
    ixes = []  # 初始化整数序列
    a = hprev  # 初始化隐藏层a
    b = hprev  # 初始化隐藏层b
    for t in range(n):  # 对序列中的每个字符
        a = sigmoid(np.dot(Ux, x) + np.dot(Wx, a) + s1)  # 计算隐藏层a
        b = sigmoid(np.dot(Vx, a) + np.dot(Rx, a) + np.dot(Tx, b) + s2)  # 计算隐藏层b
        y = softmax(np.dot(Qx, b) + s3)  # 计算输出层
        ix = np.random.choice(range(vocab_size), p=y.ravel())  # 从输出层中抽取一个整数
        x = np.zeros((vocab_size, 1))  # 初始化输入
        x[ix] = 1  # 将当前字符对应的输入置为1
        ixes.append(ix)  # 将抽取的整数添加到整数序列中
    return ixes  # 返回整数序列


"""
-------------------------------------------------------------------------------
"""

###############
###############
# Model Training #
###############
###############

# 初始化训练模型
epoch, p = 0, 0  # 初始化迭代次数和指针

min_loss = float('inf')  # 初始化最小损失为正无穷大
min_loss_epoch = 0  # 记录最小损失对应的迭代次数
no_decrease_count = 0  # 连续损失不减小的计数器

mUx, mWx, mVx, mRx, mTx, mQx = np.zeros_like(Ux), np.zeros_like(Wx), np.zeros_like(Vx), \
    np.zeros_like(Rx), np.zeros_like(Tx), np.zeros_like(Qx)  # memory variables for Adagrad
ms1, ms2, ms3 = np.zeros_like(s1), np.zeros_like(s1), np.zeros_like(s3)  # memory variables for Adagrad

smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

# 训练循环
while True:
    # 检查是否需要重置隐藏状态和数据指针
    if p + seq_length + 1 >= len(data) or epoch == 0:  # 指针到达数据末尾
        hprev = np.zeros((hidden_size, 1))  # 重置RNN的隐藏状态
        p = 0  # 回到数据起始位置

    # 从数据中提取输入和目标序列
    inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
    targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

    # 模型采样
    if epoch % 100 == 0:
        sample_ix = sample(hprev, inputs[0], 600)  # 从当前隐藏状态开始采样
        txt = ''.join(ix_to_char[ix] for ix in sample_ix)  # 将采样的序列转换为文本
        print('----\n %s \n----' % (txt,))

    # 前向传播和反向传播
    loss, dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3 = lossFun(inputs, targets)

    smooth_loss = smooth_loss * 0.999 + loss * 0.001  # 平滑损失

    if epoch % 100 == 0:
        print('epoch %d, loss: %f' % (epoch, smooth_loss))  # 打印损失

    # 参数更新
    for param, dparam, mem in zip([Ux, Wx, Vx, Rx, Tx, Qx, s1, s2, s3], [dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3]
            , [mUx, mWx, mVx, mRx, mTx, mQx, ms1, ms2, ms3]):
        mem += dparam * dparam
        param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Adagrad更新

    p += seq_length  # 移动数据指针
    epoch += 1  # 迭代计数器

    # 检查损失是否不再减小
    if smooth_loss < min_loss:  # 损失减小
        min_loss = smooth_loss  # 更新最小损失
        min_loss_epoch = epoch  # 更新最小损失对应的迭代次数
        no_decrease_count = 0  # 重置连续损失不减小的计数器
    else:
        no_decrease_count += 1  # 连续损失不减小的计数器加1
    if no_decrease_count >= 1000:  # 连续损失不减小的计数器达到20000
        break  # 停止训练

print("epoch %d, Minimum loss: %f" % (min_loss_epoch, min_loss))  # 打印最小损失对应的迭代次数和最小损失
print('----\n %s \n----' % (txt,))
"""
-------------------------------------------------------------------------------
"""
