import numpy as np
import os

#######
# Data preprocessing
#######

data=''
txt_filenames = os.listdir(r'./input')

for filename in txt_filenames:
  txt_file = open('./input/'+filename, 'r',encoding='utf-8')
  buf = txt_file.read()
  data = data+"\n"+buf
  txt_file.close()


chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
print('data has %d characters, %d unique.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(chars) }
ix_to_char = { i:ch for i,ch in enumerate(chars) }

#######
# Initializing #
#######

hidden_size = 100  # Hidden layer size
seq_length = 25  # RNN sequence length
learning_rate = 0.1  # Learning rate

# 初始化模型参数

# 初始化权重矩阵
Ux = np.random.randn(hidden_size, vocab_size)*0.01 # 输入到隐藏层a的权重
Wx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层a的权重
Vx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层b的权重
Rx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层a到隐藏层b的权重
Tx = np.random.randn(hidden_size, hidden_size)*0.01 # 隐藏层b到隐藏层b的权重
Qx = np.random.randn(vocab_size, hidden_size)*0.01 # 隐藏层b到输出的权重
# 初始化bias
s1 = np.zeros((hidden_size, 1)) # 隐藏层a的偏置
s2 = np.zeros((hidden_size, 1)) # 隐藏层b的偏置
s3 = np.zeros((vocab_size, 1)) # 输出层的偏置

#######
# Loss function
#######

def sigmoid(x):
    x = np.clip(x, -500, 500) # 防止溢出
    return 1 / (1 + np.exp(-x)) # sigmoid函数

def softmax(x):
    e_x = np.exp(x - np.max(x)) # 防止溢出
    return e_x / e_x.sum(axis=0) # softmax函数

def lossFun(inputs, targets):
  """
  inputs,targets 为整数列表.
  hprev是初始隐藏状态的hx1数组
  返回损失、模型参数的梯度和最后一个隐藏状态
  """
  x, a, b, o = {}, {}, {}, {}
  a[-1] = np.zeros((hidden_size, 1))
  b[-1] = np.zeros((hidden_size, 1))
  loss = 0
  # 前向传播
  for t in range(len(inputs)):
      x = np.zeros((vocab_size, 1))
      x[inputs[t]] = 1
      a[t] = sigmoid(np.dot(Ux, x) + np.dot(Wx, a[t - 1]) + s1)
      b[t] = sigmoid(np.dot(Vx, a[t]) + np.dot(Rx, a[t - 1]) + np.dot(Tx, b[t - 1]) + s2)
      o[t] = softmax(np.dot(Qx, b[t]) + s3)
      loss += -np.log(o[t][targets[t], 0]) # softmax (cross-entropy loss)

  # 后向传播
  dUx, dWx, dVx, dRx, dTx, dQx = np.zeros_like(Ux), np.zeros_like(Wx), np.zeros_like(Vx), \
      np.zeros_like(Rx), np.zeros_like(Tx), np.zeros_like(Qx)

  ds1, ds2, ds3 = np.zeros_like(s1), np.zeros_like(s2), np.zeros_like(s3)

  for t in reversed(range(len(inputs))):
    do = o[t].copy()
    do[targets[t]] -= 1 # backprop into o

    dQx += np.dot(do, b[t].T)
    ds3 += do

    dbt = np.dot(Qx.T, do) # backprop into h
    dbt_raw = dbt * (1 - b[t] * b[t]) # backprop through tanh nonlinearity

    dVx += np.dot(dbt_raw, a[t].T)
    dRx += np.dot(dbt_raw, a[t - 1].T)
    dTx += np.dot(dbt_raw, b[t - 1].T)
    ds2 += dbt_raw

    dat = np.dot(Vx.T, dbt_raw) + np.dot(Rx.T, dbt_raw)
    dat_raw = dat * (1 - a[t] * a[t])

    dUx += np.dot(dat_raw, x.T)
    dWx += np.dot(dat_raw, a[t - 1].T)
    ds1 += dat_raw

  for dparam in [dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3]:
    np.clip(dparam, -5, 5, out=dparam) # clip to mitigate exploding gradients

  return loss, dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3

#######
# Sampling
#######

# 定义模型的采样参数
def sample(hprev, seed_ix, n):
  """
  从模型中抽取一个整数序列。
  h是内存状态，seed_ix是第一个时间步的种子字母
  """
  x = np.zeros((vocab_size, 1))
  x[seed_ix] = 1
  ixes = []
  a = hprev
  b = hprev
  for t in range(n):
      a = sigmoid(np.dot(Ux, x) + np.dot(Wx, a) + s1)
      b = sigmoid(np.dot(Vx, a) + np.dot(Rx, a) + np.dot(Tx, b) + s2)
      y = softmax(np.dot(Qx, b) + s3)
      ix = np.random.choice(range(vocab_size), p=y.ravel())
      x = np.zeros((vocab_size, 1))
      x[ix] = 1
      ixes.append(ix)
  return ixes


#######
# Training
#######

# 初始化训练模型
epoch, p = 0, 0
mUx, mWx, mVx, mRx, mTx, mQx = np.zeros_like(Ux), np.zeros_like(Wx), np.zeros_like(Vx), \
    np.zeros_like(Rx), np.zeros_like(Tx), np.zeros_like(Qx) # memory variables for Adagrad
ms1, ms2, ms3 = np.zeros_like(s1), np.zeros_like(s1), np.zeros_like(s3) # memory variables for Adagrad

smooth_loss = -np.log(1.0/vocab_size)*seq_length # loss at iteration 0

# 训练循环
while True:
  # 检查是否需要重置隐藏状态和数据指针
  if p + seq_length + 1 >= len(data) or epoch == 0:
    hprev = np.zeros((hidden_size, 1))  # 重置RNN的隐藏状态
    p = 0  # 回到数据起始位置

  # 从数据中提取输入和目标序列
  inputs = [char_to_ix[ch] for ch in data[p:p + seq_length]]
  targets = [char_to_ix[ch] for ch in data[p + 1:p + seq_length + 1]]

  # 模型采样
  if epoch % 100 == 0:
    sample_ix = sample(hprev, inputs[0], 600)  # 从当前隐藏状态开始采样
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)  # 将采样的序列转换为文本
    print('----\n %s \n----' % (txt, ))

  # 前向传播和反向传播
  loss, dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3 = lossFun(inputs, targets)

  smooth_loss = smooth_loss * 0.999 + loss * 0.001  # 平滑损失

  if epoch % 100 == 0:
    print('iter %d, loss: %f' % (epoch, smooth_loss))  # 打印损失

  # 参数更新
  for param, dparam, mem in zip([Ux, Wx, Vx, Rx, Tx, Qx, s1, s2, s3],[dUx, dWx, dVx, dRx, dTx, dQx, ds1, ds2, ds3]
          ,[mUx, mWx, mVx, mRx, mTx, mQx, ms1, ms2, ms3]):
    mem += dparam * dparam
    param += -learning_rate * dparam / np.sqrt(mem + 1e-8)  # Adagrad更新

  p += seq_length  # 移动数据指针
  epoch += 1  # 迭代计数器