"""
    Author: huangning
    Date:2020/10/27
    Func:判断二分图
"""


class Solution:

    def __init__(self):
        self.is_bipartite = True
        # 选择一个标记颜色的记号
        init_color = -1

    def isBipartite(self, graph):
        # 给每个节点初始化一个颜色
        color_li = [-1] * len(graph)

        # 给每个节点增加是否访问的标记
        visited = [False] * len(graph)

        # 选择一个初始遍历的节点
        node_init = 0
        root_color = -1

        def traverse(curr_node, sub_graph, root_color):

            # 如果不是二分图 ，直接返回, 减少判断
            if not self.is_bipartite:
                return

            visited[curr_node] = True
            curr_node_color = -1 * root_color
            color_li[curr_node] = curr_node_color

            # 遍历邻居节点
            # 首先判断是否为空列表，如果是的话，那么直接返回
            if not sub_graph[curr_node]:
                print("当前节点是", curr_node, "邻居节点是", sub_graph[curr_node])
                return
            for neighbor in sub_graph[curr_node]:

                # 如果邻居节点没有访问过，继续深入
                if not visited[neighbor]:
                    traverse(neighbor, sub_graph, curr_node_color)
                else:
                    # 和起点比较颜色
                    if color_li[neighbor] == color_li[curr_node]:
                        self.is_bipartite = False

        # 因为图不一定是连通的 所以要遍历图中的每个节点
        for node_index in range(len(graph)):

            if not visited[node_index]:
                traverse(node_index, graph, root_color)

        one_group = []
        two_group = []
        if self.is_bipartite:
            for index, value in enumerate(color_li):
                if value == -1:
                    one_group.append(index)
                else:
                    two_group.append(index)

        return one_group, two_group, self.is_bipartite


if __name__ == "__main__":
    so = Solution()
    # graph = [[1, 3], [0, 2], [1, 3], [0, 2]]
    graph = [[4],[],[4],[4],[0,2,3]]
    a, b, c = so.isBipartite(graph)
    print(a, b, c)
