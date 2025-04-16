import numpy as np
import sys

# 用法：python eval.py query的embedding文件 label的embedding文件 top_k
# 例如：如果query的embedding文件和label的embedding文件（都是predict出来的文件）分别是query.txt和label.txt，要测top 2，那么应该执行命令：
# python eval.py query.txt label.txt 2

Q = []
L = []
len_q = []
len_l = []

fquery = open(sys.argv[1], 'r', encoding='utf-8')
flabel = open(sys.argv[2], 'r', encoding='utf-8')
for q, l in zip(fquery, flabel):
    vq = np.array([int(x) for x in q.strip().split()])
    vl = np.array([int(x) for x in l.strip().split()])
    Q.append(vq)
    L.append(vl)
    len_q.append(np.dot(vq, vq))
    len_l.append(np.dot(vl, vl))
fquery.close()
flabel.close()

top_k = int(sys.argv[3])
n_hit = 0
for i, q in enumerate(Q):
    res = []
    for j, l in enumerate(L):
        sim = np.dot(q, l) / np.sqrt(len_q[i] * len_l[j])
        res.append((sim, j))
    res.sort(reverse=True)
    res = [x[1] for x in res[ : top_k]]
    if i in res:
        n_hit += 1
print('Top %d Acc: %.3f' % (top_k, float(n_hit) / len(Q)))