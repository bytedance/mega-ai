# -*- coding: UTF-8 -*-

"""
    Author: huangning
    Date: 2021/10/05
    Func: 实现Louvain社区发现算法进行社区发现
    Note:
        1. 字符串的split()方法会额外将开头末尾的空格以及换行等符号去掉
        2. 第一轮单节点社区内部自环产生权重和为0，但是第二轮经过合并之后单节点所对应的自环产生的内部权重和不一定为0
        3. 对于真正自环情况的节点，不考虑自环边带来的社区内的权重和
"""
import matplotlib.pyplot as plt
import networkx as nx
import argparse


class Louvain:

    @classmethod
    def read_text_file(cls, input_path):
        """
        从txt中提取点和边的信息，其中txt文件格式要求为 起点 终点 边的权重
        :param input_path: 输入路径
        :return: 文件中的点边信息，并且其中必然不包括自环的边
        """
        with open(input_path, 'r') as f:
            lines = f.readlines()

        nodes = {}
        edges = []
        for line in lines:
            n = line.split()
            if not n:
                break

            # 将当前访问的节点标记为1 利用字典中key的唯一性 避免节点的重复记录
            nodes[n[0]], nodes[n[1]] = 1, 1

            # 如果当前行有3个元素，那么第三个为边的权重
            edge_weight = float(n[2]) if len(n) == 3 else 1

            # 将边列表构成 (起点，终点, 边的权重)
            edges.append((n[0], n[1], edge_weight))

        # 用连续编号给点边打标
        return relabel_process(nodes, edges)

    @classmethod
    def read_gml_file(cls, input_path):
        """
        从gml中提取点和边的信息，其中gml文件格式为 node [id xx label yy value zz] edge [source xx target yy], 其中id是整型的字符串
        :param input_path: 输入路径
        :return: 文件中的点边信息
        """
        with open(input_path, 'r') as f:
            lines = f.readlines()

        nodes = {}
        edges = []

        # 初始化当前边以及边处理的状态
        current_edge, in_edge = (-1, -1, 1), 0

        for line in lines:
            words = line.split()
            if not words:
                break

            # 若第1个元素是id，那么第2个元素就是node的ID值, 将该节点保存到nodes中
            if words[0] == 'id':
                nodes[int(words[1])] = 1

            # 若第1个元素是边的起点，那么第2个元素就是起点的ID值, 替换当前边的起点，并将边的状态标记为处理中
            elif words[0] == 'source':
                in_edge = 1
                current_edge = (int(words[1]), current_edge[1], current_edge[2])

            # 若第1个元素是边的终点 并且正在处理边 那么第2个元素就是终点的ID值
            elif words[0] == 'target' and in_edge:
                current_edge = (current_edge[0], int(words[1]), current_edge[2])

            # 若第1个元素是value 并且正在处理边 那么第二个元素就是边的权重
            elif words[0] == 'value' and in_edge:
                current_edge = (current_edge[0], current_edge[1], int(words[1]))

            # 如果第1个元素是右括号 并且边标记为正在处理中 那么接结束边的处理 那么将边保存下来 并且将边的状态标记为0并且当前边重新初始化
            elif words[0] == ']' and in_edge:
                edges.append(((current_edge[0], current_edge[1]), 1))
                current_edge, in_edge = (-1, -1, 1), 0

        nodes, edges = relabel_process(nodes, edges)
        print("%d nodes, %d edges" % (len(nodes), len(edges)))
        return nodes, edges  # 此处作了一点修改

    def __init__(self, nodes, edges):
        """
            self.m: 图的所有边权重和
            self.nodes(节点列表): 新节点名称是连续整数所构成的列表
            self.edge(边列表): 以新节点名称((起点，终点)，权重)为元素构成的列表
            self.edges_of_node: 保存点边关系
            self.real_partition : 真正的划分结果
            self.self_loop_weight: 合并前后的节点自环的权重和，只在社区节点合并的地方会更新
            self.k_i: 射入每个节点的权重和 作为元素所构成的列表，其中每个元素的索引就是节点号
            self.communities: 社区列表，将每个节点号初始化成对应的社区号社区编号
        """

        # 图中各个参数的初始化
        self.m = 0
        self.nodes = nodes
        self.edges = edges
        self.edges_of_node = {}
        self.real_partitions = []
        self.k_i = [0 for _ in nodes]
        self.communities = [node for node in nodes]

        # 更新每个节点的出/入射权重和 以及 记录点和对应边的关系(key是node：value是边构成列表)
        for curr_edge in edges:

            self.m += curr_edge[2]
            self.k_i[curr_edge[0]] += curr_edge[2]
            self.k_i[curr_edge[1]] += curr_edge[2]

            # 记录起点和对应的边
            if curr_edge[0] not in self.edges_of_node:
                self.edges_of_node[curr_edge[0]] = [curr_edge]
            else:
                self.edges_of_node[curr_edge[0]].append(curr_edge)

            # 记录终点和对应的边
            if curr_edge[1] not in self.edges_of_node:
                self.edges_of_node[curr_edge[1]] = [curr_edge]

            # 避免起点和终点相同时(也就是自环)导致边的重复添加
            elif curr_edge[0] != curr_edge[1]:
                self.edges_of_node[curr_edge[1]].append(curr_edge)

    def run_louvain(self):
        """
        运行louvain模型
        :return: 最终得到的划分结果，也就是最开始的节点号构成列表作为元素，构成的社区划分列表
        """

        # 这里self.nodes, self.edges经过最开始的预处理之后就永远保持不变，变化的是nodes和edges
        curr_nodes, curr_edges = self.nodes, self.edges

        # 初始化模块度, 以及迭代计数器
        best_q, iter_step = -1, 1

        # 针对自环情况的节点初始化，后续每次执行第二步合并节点之后也会针对"自环"进行初始化
        self.self_loop_weight = [0 for _ in curr_nodes]
        for curr_edge in curr_edges:
            # 后续用于计算社区内权重和时，针对自环情况，计算入射权重和
            if curr_edge[0] == curr_edge[1]:
                self.self_loop_weight[curr_edge[0]] += curr_edge[2]

        tttpartition = [[node] for node in curr_nodes]
        # 单独赋值，避免出现深拷贝的情况
        self.s_in = [value*2 for value in self.self_loop_weight]
        self.s_tot = [self.k_i[node] for node in curr_nodes]
        tq = self.compute_modularity(tttpartition)
        print("初始模块度为", tq)

        # 这个地方迭代次数，设置一个默认值好一点
        while True:

            # 找到节点的最优分割方式并计算对应的模块度，其中partitions中每个元素的索引是社区号，元素值就是有节点号作为元素构成的列表
            partitions = self.first_phase(curr_nodes, curr_edges)
            q = self.compute_modularity(partitions)
            print("划分为:", partitions, "社区内权重和：", self.s_in, "与社区相连的权重和：", self.s_tot, "整张图权重和：", self.m,"当前计算模块度：", q, "历史最优模块度：", best_q)

            # 如果本轮迭代模块度没有提升，则认为收敛并停止迭代
            if q == best_q:
                break
            best_q = q

            # 将社区最优分割中的空社区集合剔除
            partitions = [c for c in partitions if c]

            if self.real_partitions:
                real_partitions = []
                for partition in partitions:
                    update_partition = []

                    # 遍历每个社区划分中的节点号，此时的partition中的每个节点编号仍然等于此次迭代时对应的社区号
                    for new_node in partition:

                        # 根据本轮的node编号, 即上轮第二步构造的新节点号(新社区编号)在上一轮的real_partition中寻找原始划分并进行合并
                        update_partition.extend(self.real_partitions[new_node])

                    real_partitions.append(update_partition)
                self.real_partitions = real_partitions

            # 第一次迭代中经过first_phase后构成的分区，就是没有经过节点折叠的分区
            else:
                self.real_partitions = partitions

            print("处理后的真实划分为", self.real_partitions)

            # 如果模块度Q有提升，那么开始第二步，即社区内节点的折叠
            curr_nodes, curr_edges = self.second_phase(curr_edges, partitions)
            iter_step += 1

        # 最后返回最终的分割方法，以及对应最优的模块度
        return self.real_partitions, best_q

    def first_phase(self, nodes, edges):
        """
        # make initial partition
        # 先初始化一个社区结构，就是一个列表，列表中每个元素是一个用列表表示的社区结构
        """

        # 单独赋值，避免出现深拷贝的情况
        self.s_in = [value*2 for value in self.self_loop_weight]
        self.s_tot = [self.k_i[node] for node in nodes]
        best_partitions = [[node] for node in nodes]

        print("----------------第一步骤迭代开始----------------")
        while True:

            improvement = False
            for node in nodes:

                # 读取当前节点所在的社区号
                node_curr_community = self.communities[node]

                # 默认当前所在社区编号是最优社区编号
                node_best_community = node_curr_community

                # 初始化模块度增益为0
                best_modularity_gain = 0

                # 当前节点所在社区内 所有与该节点相连边权重和
                old_shared_links = 0

                # 从点边关系映射中 遍历当前节点 对应的所有边
                for edge in self.edges_of_node[node]:

                    # 如果当前边是自环时，对应的内部权重和算到了self_loop_weight，每一轮迭代都是固定的
                    if edge[0] == edge[1]:
                        continue

                    # 针对边的起点和终点在同一社区的情况，累加所有权重和，用于后续计算移除该节点后，社区内部边的权重和
                    if edge[0] == node and self.communities[edge[1]] == node_curr_community or edge[1] == node and self.communities[edge[0]] == node_curr_community:
                        old_shared_links += edge[2]

                neighbor_communities = {}

                # 增加是否存在增益的移动
                move_flag = False
                new_shared_links = 0
                for neighbor in self.get_neighbors(node):

                    # 获取邻居节点所在社区号
                    neighbor_community = self.communities[neighbor]

                    # 如果该邻居所在社区之前分析过，那么直接跳过
                    if neighbor_community in neighbor_communities:
                        # print("当前邻居所在的社区之前计算过对应的社区内权重和等指标")
                        continue

                    # 针对和当前节点 和邻居所在的社区 在同一个社区的情况也不需要考虑，因为移动后对应的模块度增益也为0
                    if neighbor_community == node_curr_community:
                        continue

                    neighbor_communities[neighbor_community] = 1
                    shared_links = 0
                    for edge in self.edges_of_node[node]:

                        # 针对自环情况不考虑节点的社区移动
                        if edge[0] == edge[1]:
                            continue

                        # 根据邻居所在的社区，在邻接关系中，找到所有与当前节点的邻居所在同一社区的邻居节点，求对应的权重和作为后续的k_i,in这个值
                        if (edge[0] == node and self.communities[edge[1]] == neighbor_community) or (edge[1] == node and self.communities[edge[0]] == neighbor_community):
                            shared_links += edge[2]

                    # 如果模块度增益比当前模块度增益大的话，更新社区以及增益, 增益一同考虑移除的社区和新增节点的社区
                    tmp_modularity_gain = (2 * (shared_links + self.self_loop_weight[node]) - self.s_tot[neighbor_community] * self.k_i[node] / self.m)/(2*self.m)
                    curr_modularity_gain = (2 * (shared_links - old_shared_links) - self.k_i[node] ** 2 / self.m + self.k_i[node] * (self.s_tot[node_curr_community] - self.s_tot[neighbor_community])/self.m)/(2*self.m)
                    print("作者计算增益: ", tmp_modularity_gain, "自己计算增益", curr_modularity_gain, "剔除的权重和", old_shared_links, "新增的权重和", shared_links, "tot", self.s_tot[node_curr_community], "totneightbor", self.s_tot[neighbor_community],"k_i:", self.k_i[node], "当前节点: ", node, "邻居节点: ", neighbor)
                    if curr_modularity_gain > best_modularity_gain:
                        best_modularity_gain = curr_modularity_gain
                        new_shared_links = shared_links  # 赋予之前剔除当前节点时，所在社区内部的权重和
                        node_best_community = neighbor_community
                        move_flag = True

                # 找到最优的社区划分方法之后(有可能最优的就是自身所在的社区)，就把节点加入对应的最优社区中来最大化总模块度增益

                # 该节点 存在 有增益的社区移动
                if move_flag:

                    # 找到最优的邻居社区后，更新当前节点所在的社区编号, 以及社区内权重和以及与当前社区内所有节点的入射权重和
                    best_partitions[node_curr_community].remove(node)
                    best_partitions[node_best_community].append(node)

                    # 更新当前节点所在社区编号
                    self.communities[node] = node_best_community

                    # 从原来社区移除后 旧社区内权重和以及社区之间权重和更新
                    self.s_in[node_curr_community] -= 2 * (old_shared_links + self.self_loop_weight[node])
                    self.s_tot[node_curr_community] -= self.k_i[node]

                    # 加入新社区后 新社区内权重和以及社区之间权重和进行更新
                    self.s_in[node_best_community] += 2 * (new_shared_links + self.self_loop_weight[node])
                    self.s_tot[node_best_community] += self.k_i[node]

                print("*************模块度开始计算******************")
                print("-------当前节点号：", node, "处理后历史最大模块度增益", best_modularity_gain, "best_partition", best_partitions, "当前整体模块度为", self.compute_modularity(best_partitions))

                # 如果遍历完当前节点的所有邻居社区之后，如果所在社区发生变化，就做一个标记
                if node_curr_community != node_best_community:
                    improvement = True

            # 如果遍历所有节点之后没有提升就跳出循环，否则继续重新遍历所有节点
            print("遍历完一次所有节点之后对应的划分为", best_partitions, "是否有过提升", improvement, "当前划分", best_partitions)
            if not improvement:
                break
        print("-------------------------当前first phase结束-------------------------------------")
        return best_partitions

    def second_phase(self, old_edges, partitions):
        """
        将划分中的每个社区节点都合并成新的点, 并更新对应的自环权重和
        :param old_edges: 边的集合
        :param partitions: 社区划分
        :return:
        """

        # 将first_phase得到的划分结果进行折叠，构造新节点
        new_nodes = [new_node for new_node in range(len(partitions))]

        # 给社区列表self.communities  构建新的标签映射
        communities_old2new = {}
        new_communities = []
        new_community = 0

        for old_community in self.communities:

            # 给老社区列表中每个元素构建新标签映射，如果之前构建过，直接用映射后的值来添加到社区列表中
            if old_community in communities_old2new.keys():
                new_communities.append(communities_old2new[old_community])
            else:
                # 给每个社区重新构建映射标签，并且构建新的社区列表communities_
                communities_old2new[old_community] = new_community
                new_communities.append(new_community)
                new_community += 1

        # 更新社区列表中社区的编号，但"列表长度仍然和老节点个数一样"，即当前还没有折叠
        self.communities = new_communities

        # 开始折叠操作: 即构建新边和新的节点，其中将社区合并成新节点后，社区之间边的权重和作为新节点之间边的权重
        new_edges = {}

        for old_edge in old_edges:

            # communities列表的索引对应每个旧节点编号，对应值就所在社区的新编号
            new_ci = self.communities[old_edge[0]]
            new_cj = self.communities[old_edge[1]]
            if (new_ci, new_cj) in new_edges.keys():
                new_edges[(new_ci, new_cj)] += old_edge[2]
            else:
                new_edges[(new_ci, new_cj)] = old_edge[2]
        new_edges = [(edge[0], edge[1], edge_weight) for edge, edge_weight in new_edges.items()]

        # 重新初始化 整个图的权重和，新节点的入射权重, k_i以及新的edges_of_node点边关系(新节点数量要小于等于老节点的个数)
        self.k_i = [0 for _ in new_nodes]
        self.self_loop_weight = [0 for _ in new_nodes]
        self.edges_of_node = {}
        for new_edge in new_edges:

            # 重构后更新新图的权重和
            self.k_i[new_edge[0]] += new_edge[2]
            self.k_i[new_edge[1]] += new_edge[2]

            # 在社区节点合并后，更新"自环"的边权重
            if new_edge[0] == new_edge[1]:
                self.self_loop_weight[new_edge[0]] += new_edge[2]

            # 保存对应的点边关系
            if new_edge[0] not in self.edges_of_node.keys():
                self.edges_of_node[new_edge[0]] = [new_edge]
            else:
                self.edges_of_node[new_edge[0]].append(new_edge)

            if new_edge[1] not in self.edges_of_node.keys():
                self.edges_of_node[new_edge[1]] = [new_edge]

            # 避免自环情况下，重复添加两次同一个边
            elif new_edge[0] != new_edge[1]:
                self.edges_of_node[new_edge[1]].append(new_edge)

        # 用折叠后的节点来更新社区额列表，其中列表索引就是新节点编号，列表元素值就是节点所在的社区值
        self.communities = new_nodes
        return new_nodes, new_edges

    def get_neighbors(self, node):
        """
        寻找所有的邻居节点
        :param node:
        :return:
        """

        for e in self.edges_of_node[node]:

            # 自环节点的邻居一定和节点本身在同一社区
            if e[0] == e[1]:
                continue
            if e[0] == node:
                yield e[1]
            if e[1] == node:
                yield e[0]

    def compute_modularity(self, partitions):
        """
        计算整个网络的模块度，即将每个社区划分的模块度进行累加
        :param partitions: 划分的社区
        :return: 整个网络的模块度Q
        """
        q = 0
        for i in range(len(partitions)):
            q += self.s_in[i] / (self.m * 2) - (self.s_tot[i] / (self.m * 2)) ** 2
        return q


def relabel_process(nodes, edges):
    """
    用连续的ID来标记点, 进而重新构建边和所在的图
    :param nodes: {nodeA: 1,nodeB: 1}
    :param edges: ((nodeA, nodeB), weight)
    :return: 重新标记的点和边
    """

    nodes = list(nodes.keys())
    nodes_, edges_ = [], []
    nodes_name2index = {}

    # 构建节点旧名称和新名称的映射，新名称就是从零开始的索引
    for new_node, old_node in enumerate(nodes):
        nodes_.append(new_node)
        nodes_name2index[old_node] = new_node

    # 构建边，用新的点名称以及原始的边权重
    for edge in edges:
        edges_.append((nodes_name2index[edge[0]], nodes_name2index[edge[1]], edge[2]))

    print("真实节点和重新标节点映射关系为")
    print(nodes_name2index)
    return nodes_, edges_


def draw_network(G, partitions):

    # 将节点标号都+1
    for i in range(len(partitions)):
        for j in range(len(partitions[i])):
            partitions[i][j] += 1

    color_list = []
    for color_index, partition in enumerate(partitions):

        # 获取一个随机的颜色, 用于当前社区划分中
        color_index = (color_index+25) % 255
        for node in partition:
            color_list.append((node, color_index))

    # 针对每组社区划分中，针对节点号进行排序
    color_list.sort(key=lambda x: x[0])

    # 将所有的颜色编号取出来
    color_list = [x[1] for x in color_list]

    plt.figure(figsize=(5, 5))
    pos = nx.spring_layout(G)
    nx.draw(G, with_labels=True, node_color=color_list, pos=pos)
    plt.show()


def main():
    print("开始测试Louvain模型...")

    # 设置参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str,
                        default="/Users/huangning/ByteCode/magellan_ai/data/graph_demo.txt",
                        help="input_path is the path of the net")

    # 解析参数
    args = parser.parse_args()
    input_path = args.input_path

    # 不同格式文件的读入方法
    if input_path[-3:] == "txt":
        net_nodes, net_edges = Louvain.read_text_file(input_path)

    elif input_path[-3:] == "gml":
        net_nodes, net_edges = Louvain.read_gml_file(input_path)

    print("图中共有{}个点，{}个边".format(len(net_nodes), len(net_edges)))

    net_louvain = Louvain(net_nodes, net_edges)
    best_partitions, best_q = net_louvain.run_louvain()
    print("最优分割为", best_partitions)

    # 将点和边转成图
    G = nx.Graph()
    G.add_nodes_from(net_nodes)

    # 添加有权重的边
    new_edge = [(edge[0], edge[1], edge[2]) for edge in net_edges]
    G.add_weighted_edges_from(new_edge)
    draw_network(G, best_partitions)


if __name__ == "__main__":
    main()
