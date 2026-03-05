import os
import time
import torch
import argparse

from model import SASRec
from utils import *
import csv
import pandas as pd
from tqdm import tqdm

data  = pd.read_csv('./data/ml-1m.txt', header=None)
print('lll')
print(data)

class args():
    def __init__(self):
        self.dataset = 'ml-1m'
        self.batch_size = 128
        self.lr = 0.001
        self.maxlen = 200
        self.hidden_units = 50
        self.num_blocks = 2
        self.num_epochs = 201
        self.num_heads  = 1
        self.dropout_rate = 0.2
        self.l2_emb = 0.0
        self.inference_only = False
        self.state_dict_path = None
        self.device = 'cpu'

args = args()


usernum = 0
itemnum = 0
User = defaultdict(list)
user_train = {}
user_valid = {}
user_test = {}
# assume user/item index starting from 1
f = open('data/%s.txt' % 'ml-1m', 'r')
print('llll')
for line in f:
    u, i = line.rstrip().split(',')
    u = int(u)
    i = int(i)
    usernum = max(u, usernum)
    itemnum = max(i, itemnum)
    User[u].append(i)


model = SASRec(53424, 10000, args)
model.load_state_dict(torch.load('./ml-1m_default/SASRec.epoch=201.lr=0.001.layer=2.head=1.hidden=50.maxlen=200.pth', map_location=torch.device(args.device)))
model.eval()
user_test = defaultdict(list)
print('ppp')
for i in tqdm(range(1, usernum+1)):
    j = list(range(1, itemnum+1))
    user_test[i].append(list(set(j).difference(User[i])))
    user_test[i] = user_test[i][0]

with open('./sub_f.csv', 'ab') as f:
    print('kkk')
    maxlen_te = max(len(user_test[i]) for i in range(1, usernum+1))
    maxlen_tr = max(len(User[i]) for i in range(1, usernum+1))
    for i in tqdm(range(1, usernum+1)):
        seq = np.zeros([maxlen_tr], dtype=np.int32)
        idx = maxlen_tr - 1
        for j in reversed(User[i]):
            seq[idx] = j
            idx -= 1
            if idx == -1: break
        item_idx = user_test[i]
        p = [np.array(l) for l in [[i], [seq], item_idx]]
        predictions = -model.predict(*p)
        predictions = predictions[0]
        a = predictions.argsort()[:10]
        a = a.numpy()
        r = np.array(item_idx)
        s = r[a]
        u = np.full(shape=10, fill_value=i, dtype=np.int64)
        pre = np.c_[u-1,s-1]
        np.savetxt(f, pre, delimiter=',', fmt='%i')
df = pd.read_csv('./sub_f.csv',header=None,names=['user_id', 'item_id'])
df.to_csv('./submission_1.csv',index=False)