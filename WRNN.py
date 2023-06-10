import numpy as np
import os

###############
###############
# Data Preprocessing #
###############
###############

# 获取指定路径 './input' 下的所有文件名，并将其存储在 txt_filenames 列表中
data=''
txt_filenames = os.listdir(r'./input')

for filename in txt_filenames:# 使用 for 循环遍历 txt_filenames 列表中的每个文件名
  txt_file = open('./input/'+filename, 'r',encoding='utf-8') # 读取 txt_file 中的所有文本，并将结果赋值给变量 buf
  buf = txt_file.read() # 读取 txt_file 中的所有文本，并将结果赋值给变量 buf
  data = data+"\n"+buf # 将 buf 中的文本添加到 data 中
  txt_file.close()  # 关闭 txt_file 文件


chars = list(set(data)) # 输出 data 的数据类型
data_size, vocab_size = len(data), len(chars) # 输出 chars的长度
print('data has %d characters, %d unique.' % (data_size, vocab_size)) # 输出 data 的长度和 chars 的长度

char_to_ix = { ch:i for i,ch in enumerate(chars) } # 将 chars 中的字符转换为索引 index，并将结果存储在字典 char_to_ix 中
ix_to_char = { i:ch for i,ch in enumerate(chars) } # 将 chars 中的索引 index 转换为字符，并将结果存储在字典 ix_to_char 中

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Model Initializing #
###############
###############

# 模型超参数（要修改的话，请修改这里 by Gu Rui）
hidden_size = 100  # Hidden layer size
seq_length = 25  # RNN sequence length
learning_rate = 0.1  # Learning rate

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
Ux = np.random.randn(hidden_size, vocab_size)*0.01 # 输入到隐藏层a的权重
Wx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层a的权重
Vx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层b的权重
Rx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层b的权重
Tx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层b到隐藏层b的权重
Qx = np.random.randn(vocab_size, hidden_size)*0.01 # 隐藏层b到输出的权重

s1 = np.zeros((hidden_size, 1)) # 隐藏层a的偏置
s2 = np.zeros((hidden_size, 1)) # 隐藏层b的偏置
s3 = np.zeros((vocab_size, 1)) # 输出层的偏置

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Loss Function #
###############
###############

def sigmoid(x):
    x = np.clip(x, -500, 500) # 防止溢出
    return 1 / (1 + np.exp(-x)) # sigmoid函数

def softmax(x):
    e_x = np.exp(x - np.max(x)) # 防止溢出
    return e_x / e_x.sum(axis=0) # softmax函数

"""
-------------------------------------------------------------------------------
"""