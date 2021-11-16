import numpy as np
"""
    Author: huangning
    Date: 2020.09.27
    Note: 实现node2vec模型

"""


# 初始生成节点到节点的概率
def preprocess_transition_probs(self):
    """
        为了后续的的随机游走，生成转移概率矩阵
    """

    ## get_alias_edge函数 是 对每条边 设定为二阶 randomwalk 的概率形式
    ## 这个函数的作用是生成每个边界的概率，同时会有alias_setup这个函数将概率进行转换，方便后面抽样


    G = self.G                         # 将图对象赋值
    is_directed = self.is_directed     # 是否是无向图

    alias_nodes = {}
    for node in G.nodes():

        unnormalized_probs = [G[node][nbr]['weight'] for nbr in sorted(G.neighbors(node))]  # 读取每个邻居节点权重
        norm_const = sum(unnormalized_probs)                                                # 权重求和，作为公式中正则项常数的那个分母
        normalized_probs = [float(u_prob)/norm_const for u_prob in unnormalized_probs]      # 除以分母获取归一化的转移概率
        alias_nodes[node] = alias_setup(normalized_probs)


def alias_setup(probs):
    """
     alias_setup 的作用是根据 二阶random walk输出的概率变成每个节点对应两个数，被后面的alias_draw函数所进行抽样
    """

    K = len(probs)
    q = np.zeros(K)
    J = np.zeros(K, dtype=np.int)

    smaller = []
    larger = []
    for kk, prob in enumerate(probs):
        q[kk] = K*prob
        if q[kk] < 1.0:
            smaller.append(kk)
        else:
            larger.append(kk)  # kk是下标，表示哪些下标小

    while len(smaller) > 0 and len(larger) > 0:
        small = smaller.pop()  # smaller自己也会减少最右边的值
        large = larger.pop()

        J[small] = large
        q[large] = q[large] + q[small] - 1.0
        if q[large] < 1.0:
            smaller.append(large)
        else:
            larger.append(large)

    return J, q




if __name__ == "__main__":
    print("开始实现node2vec")