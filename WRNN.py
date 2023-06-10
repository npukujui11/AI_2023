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
