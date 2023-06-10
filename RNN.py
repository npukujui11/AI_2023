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
# Loss Function #
###############
###############

def lossFun(inputs, targets, hprev):
  """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
  """
  xs, hs, ys, ps = {}, {}, {}, {}
  hs[-1] = np.copy(hprev)
  loss = 0
  # 前向传播
  for t in range(len(inputs)):
    xs[t] = np.zeros((vocab_size,1)) # encode in 1-of-k representation
    xs[t][inputs[t]] = 1
    hs[t] = np.tanh(np.dot(Wxh, xs[t]) + np.dot(Whh, hs[t-1]) + bh) # hidden state
    ys[t] = np.dot(Why, hs[t]) + by # unnormalized log probabilities for next chars
    ps[t] = np.exp(ys[t]) / np.sum(np.exp(ys[t])) # probabilities for next chars
    loss += -np.log(ps[t][targets[t],0]) # softmax (cross-entropy loss)
  # 反向传播
  dWxh, dWhh, dWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
  dbh, dby = np.zeros_like(bh), np.zeros_like(by)
  dhnext = np.zeros_like(hs[0])
  for t in reversed(range(len(inputs))):
    dy = np.copy(ps[t])
    dy[targets[t]] -= 1 # backprop into y
    dWhy += np.dot(dy, hs[t].T)
    dby += dy
    dh = np.dot(Why.T, dy) + dhnext # backprop into h
    dhraw = (1 - hs[t] * hs[t]) * dh # backprop through tanh nonlinearity
    dbh += dhraw
    dWxh += np.dot(dhraw, xs[t].T)
    dWhh += np.dot(dhraw, hs[t-1].T)
    dhnext = np.dot(Whh.T, dhraw)
  for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients
  return loss, dWxh, dWhh, dWhy, dbh, dby, hs[len(inputs)-1]

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Result Sampling #
###############
###############

# 定义模型的采样参数
def sample(h, seed_ix, n):
  """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  for t in range(n):
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h) + bh)
    y = np.dot(Why, h) + by
    p = np.exp(y) / np.sum(np.exp(y))
    ix = np.random.choice(range(vocab_size), p=p.ravel())
    x = np.zeros((vocab_size, 1))
    x[ix] = 1
    ixes.append(ix)
  return ixes

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Hyperparameters #
###############
###############

# 模型超参数（要修改的话，请修改这里 by Gu Rui）
hidden_sizes = range(50, 1001, 50)  # Hidden layer size
seq_lengths = range(25, 101, 10)  # RNN sequence length
learning_rates = [1e-2, 1e-3, 1e-4, 1e-5]  # Learning rate

results = []  # 保存每次训练的结果

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Model Training #
###############
###############

for hidden_size in hidden_sizes:
    for seq_length in seq_lengths:
        for learning_rate in learning_rates:
            # 初始化模型参数
            # 随机初始化这些参数是为了打破对称性，并且乘以一个小的常数（例如0.01）是为了保持参数的相对较小的初始值，以帮助模型更快地收敛。
            Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # 输入到隐藏层的权重
            Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # 隐藏层到隐藏层的权重
            Why = np.random.randn(vocab_size, hidden_size) * 0.01  # 隐藏层到输出层的权重
            bh = np.zeros((hidden_size, 1))  # 隐藏层的偏置
            by = np.zeros((vocab_size, 1))  # 输出层的偏执

            # 初始化训练模型
            n, p = 0, 0

            min_loss = float('inf')  # 初始化最小损失为正无穷大
            min_loss_epoch = 0  # 记录最小损失对应的迭代次数
            no_decrease_count = 0  # 连续损失不减小的计数器
            int_no_decrease_count = 0  # 连续整数位没有减少的计数器no_decrease_count = 0  # 连续整数位没有减少的计数器
            prev_loss = None  # 上一个epoch的整数位损失函数值

            mWxh, mWhh, mWhy = np.zeros_like(Wxh), np.zeros_like(Whh), np.zeros_like(Why)
            mbh, mby = np.zeros_like(bh), np.zeros_like(by)  # memory variables for Adagrad
            smooth_loss = -np.log(1.0 / vocab_size) * seq_length  # loss at iteration 0

            # 训练循环
            while True:
                # 检查是否需要重置隐藏状态和数据指针
                if p + seq_length + 1 >= len(data) or n == 0:
                    hprev = np.zeros((hidden_size, 1))  # 重置RNN的隐藏状态
                    p = 0  # 回到数据起始位置

                # 从数据中提取输入和目标序列
                inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
                targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

                # 模型采样
                if n % 100 == 0:
                    sample_ix = sample(hprev, inputs[0], 600)  # 从当前隐藏状态开始采样
                    txt = ''.join(ix_to_char[ix] for ix in sample_ix)  # 将采样的序列转换为文本
                    print('----\n %s \n----' % (txt,))

                # 前向传播和反向传播
                loss, dWxh, dWhh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
                smooth_loss = smooth_loss * 0.999 + loss * 0.001  # 平滑损失
                if n % 100 == 0:
                    print('iter %d, loss: %f' % (n, smooth_loss))  # 打印损失

                # 参数更新
                for param, dparam, mem in zip([Wxh, Whh, Why, bh, by], [dWxh, dWhh, dWhy, dbh, dby],
                                              [mWxh, mWhh, mWhy, mbh, mby]):
                    mem += dparam * dparam
                    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Adagrad更新

                p += seq_length  # 移动数据指针
                n += 1  # 迭代计数器

                # 检查损失是否不再减小
                if smooth_loss < min_loss:  # 损失减小
                    min_loss = smooth_loss  # 更新最小损失
                    min_loss_epoch = n  # 更新最小损失对应的迭代次数
                    no_decrease_count = 0  # 重置连续损失不减小的计数器
                else:
                    no_decrease_count += 1  # 连续损失不减小的计数器加1
                if no_decrease_count >= 1000:  # 连续损失不减小的计数器达到20000
                    break  # 停止训练

                # 判断是否满足终止条件
                if int_no_decrease_count >= 10000:
                    break

                # 检查整数位损失函数是否减少
                if prev_loss is not None and int(smooth_loss) >= int(prev_loss):
                    no_decrease_count += 1
                else:
                    no_decrease_count = 0

                prev_loss = smooth_loss

            # 保存结果
            result = {
                'hidden_size': hidden_size,
                'seq_length': seq_length,
                'learning_rate': learning_rate,
                'min_loss': min_loss,
                'min_loss_epoch': min_loss_epoch,
                'txt': txt
            }
            results.append(result)

"""
-------------------------------------------------------------------------------
"""

###############
###############
# Model Evaluation #
###############
###############

# 输出结果
for result in results:
    print("Hidden Size:", result['hidden_size'])
    print("Seq Length:", result['seq_length'])
    print("Learning Rate:", result['learning_rate'])
    print("Minimum Loss:", result['min_loss'])
    print("Epoch:", result['min_loss_epoch'])
    print("Text:", result['txt'])
    print("--------------------")

"""
-------------------------------------------------------------------------------
"""