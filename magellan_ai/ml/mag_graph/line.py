import networkx as nx
"""
    Author: huangning
    Date: 2021/04/09
    Target: 构建Line模型
"""

class LINE:

    def __init__(self, graph, embedding_size=8, negative_ratio=5, order='second',):
        """

        :param graph:
        :param embedding_size:
        :param negative_ratio:
        :param order: 'first','second','all'
        """

        # 判断是一阶相似度，二姐相似度以及
        if order not in ['first', 'second', 'all']:
            raise ValueError('mode must be fisrt,second,or all')


if __name__ == "__main__":

    print("开始创建LINE模型")

    # 从txt文件中读入有向图
    G = nx.read_edgelist("../data/Wiki_edgelist.txt", create_using=nx.DiGraph(), nodetype=None, data=[('weight', int)])
    print(type(G))

    model = LINE(G, embedding_size=7, order='second')