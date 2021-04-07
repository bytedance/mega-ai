# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 09:57:22 2019

@author: muli
"""

import scipy.io as scio

path = 'blogcatalog.mat'
data = scio.loadmat(path)
# 查看mat文件的数据格式
# # <class 'dict'>
# print(type(data))
#
# # 查看字典的键
# # dict_keys(['__header__', '__version__', '__globals__', 'bags', 'targets', 'class_name'])
# print(data.keys())

# dict_keys(['__header__', '__version__', '__globals__', 'X', 'y'])

# 选择需要的数据；数组格式
from scipy.sparse import issparse
x = data['network']
print("====")
print(issparse(x))
y = x.todense()
print(issparse(y))

print(x.tocoo())
print("----------------")
# for i in x.tocoo:
#     print()
cx = x.tocoo()
print(cx.row, cx.col, cx.data)
c = {"1":[2,"1"], "3":[4]}
print(list(c))

from six import iterkeys

for i in iterkeys(c):
    if i in c[i]:
        c[i].remove(i)
print(c)
# import numpy as np
# import scipy.sparse as ss
#
# a = np.zeros((3, 4))
# a[1, 2] = 12
# a[2, 2] = 22
# print(a)
# print(ss.csc_matrix(a))
#
# x = ss.csc_matrix((4, 3))
# # x = ss.lil_matrix((4, 3))
# print("x --")
# print(x)
# print("====")
#
# x[1, 2] = 12
# x[3, 1] = 23
#
# print(x)
# print(x.todense())

# import numpy as np
#
# import networkx as nx
#
# np.save('network.adj', x)
#
# # -------DIRECTED Graph, Unweighted-----------
# # Unweighted directed graph:
# a = np.loadtxt('network.adj.npy', delimiter=' ', dtype=int)
# D = nx.DiGraph(a)
# nx.write_edgelist(D, 'network.edgelist', data=False)  # output

# with open('nju_labels.txt', mode='a') as f:
#     for i in range(2000):
#         b = x[:, i]
#         for j in range(5):
#             if b[j] == 1:
#                 print(j)
#                 f.write(str(j)+" ")
#         f.write("\n")
#         print("-----------------------")
