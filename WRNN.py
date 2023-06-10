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