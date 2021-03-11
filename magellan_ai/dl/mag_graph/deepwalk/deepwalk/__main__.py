#!/usr/bin/env python
# -*- coding: utf-8 -*-

from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from multiprocessing import cpu_count
from gensim.models import Word2Vec
from skipgram import Skipgram
from six.moves import range

import walks as serialized_walks
import logging
import psutil
import random
import graph
import sys
import os

"""
    Author: huangning.honey
    Date: 2021/03/09
    Func:
    word2vec参数说明
    1. sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出，下面演示的就是从文件中读出
    2. size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度
    3. window：即词向量上下文最大距离，这个参数在我们的算法原理篇中标记为𝑐，window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间
    4. sg: 即我们的word2vec两个模型的选择了。如果是0，则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。
    5. hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。
    6. negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。这个参数在我们的算法原理篇中标记为neg。
    7. cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的𝑥𝑤为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示𝑥𝑤,默认值也是1,不推荐修改默认值
    8. min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值
    9. iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值
    10. alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为𝜂，默认是0.025
    11. min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。
    这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值
"""

p = psutil.Process(os.getpid())
try:
    p.set_cpu_affinity(list(range(cpu_count())))
except AttributeError:
    try:
        p.cpu_affinity(list(range(cpu_count())))
    except AttributeError:
        pass

logger = logging.getLogger(__name__)
LOGFORMAT = "%(asctime).19s %(levelname)s %(filename)s: %(lineno)s %(message)s"


# debug处理
def debug(type_, value, tb):
    if hasattr(sys, 'ps1') or not sys.stderr.isatty():
        sys.__excepthook__(type_, value, tb)
    else:
        import traceback
        import pdb
        traceback.print_exception(type_, value, tb)
        print(u"\n")
        pdb.pm()


def process(args):

    # 根据数据的不同格式加载数据表
    if args.format == "adjlist":
        G = graph.load_adjacencylist(args.input, undirected=args.undirected)
    elif args.format == "edgelist":
        G = graph.load_edgelist(args.input, undirected=args.undirected)
    elif args.format == "mat":
        G = graph.load_matfile(args.input, variable_name=args.matfile_variable_name, undirected=args.undirected)
    else:
        raise Exception("Unknown file format: '%s'.  Valid formats: 'adjlist', 'edgelist', 'mat'" % args.format)

    print("节点数量: {}".format(len(G.nodes())))
    num_walks = len(G.nodes()) * args.number_walks
    print("游走数量: {}".format(num_walks))
    data_size = num_walks * args.walk_length
    print("总的节点数量: {}".format(data_size))

    # 当训练的总节点数量小于输入磁盘的阈值
    if data_size < args.max_memory_data_size:

        print("随机游走生成中...")
        walks = graph.build_deepwalk_corpus(G, num_paths=args.number_walks, path_length=args.walk_length, alpha=0,
                                            rand=random.Random(args.seed))

        print("模型训练中...")
        model = Word2Vec(sentences=walks, size=args.representation_size, window=args.window_size,
                         sg=1, hs=1, negative=5, min_count=0, workers=args.workers,
                         iter=5, alpha=0.025, min_alpha=0.01)

    else:
        print("Data size {} is larger than limit (max-memory-data-size: {}). Dumping walks to disk.".format(data_size, args.max_memory_data_size))
        print("随机游走生成中...")

        walks_filebase = args.output + ".walks"
        walk_files = serialized_walks.write_walks_to_disk(G, walks_filebase, number_walks=args.number_walks,
                                                          path_length=args.walk_length, alpha=0,
                                                          rand=random.Random(args.seed),
                                                          number_workers=args.workers)

        print("Counting vertex frequency...")
        if not args.vertex_freq_degree:
            vertex_counts = serialized_walks.count_textfiles(walk_files, args.workers)
        else:
            # use degree distribution for frequency in tree
            vertex_counts = G.degree(nodes=G.iterkeys())

        print("Training...")
        walks_corpus = serialized_walks.WalksCorpus(walk_files)
        model = Skipgram(sentences=walks_corpus, vocabulary_counts=vertex_counts,
                         size=args.representation_size,
                         window=args.window_size, min_count=0, trim_rule=None, workers=args.workers)

    model.wv.save_word2vec_format(args.output)


def deepwalk_entry(input_path, output_path):
    """
    ArgumentParser.add_argument 参数说明
    1. nargs: 设置参数可以提供的个数, x的候选值如下
        N   参数的绝对个数（例如：3）
        '?'   0或1个参数
        '*'   0或所有参数
        '+'   所有，并且至少一个参数
    2. required: 表示这个参数是否一定需要设置
        如果设置了required=True,则在实际运行的时候不设置该参数将报错
    3. default：没有设置值情况下的默认参数
        default表示命令行没有设置该参数的时候，程序中用什么值来代替
    4. help：指定参数的说明信息
        在现实帮助信息的时候，help参数的值可以给使用工具的人提供该参数是用来设置什么的说明，对于大型的项目，help参数和很有必要的，不然使用者不太明白每个参数的含义，增大了使用难度。
    5. type：参数类型
        默认的参数类型是str类型，如果你的程序需要一个整数或者布尔型参数，你需要设置type=int或type=bool，下面是一个打印平方的例子
    6. choices：参数值只能从几个选项里面选择
    7. dest：设置参数在代码中的变量名
        argparse默认的变量名是--或-后面的字符串，但是你也可以通过dest=xxx来设置参数的变量名，然后在代码中用args.xxx来获取参数的值。
    """

    parser = ArgumentParser("huangning-deepwalk", formatter_class=ArgumentDefaultsHelpFormatter
                            , conflict_handler='resolve')

    parser.add_argument('--input', default=input_path, nargs='?', help='Graph样本的输入路径')
    parser.add_argument('--output', default=output_path, help='表示文件的输出路径')

    parser.add_argument('--max-memory-data-size', default=10000000, type=int, help='将walk导入到磁盘的大小，代替放入内存中.')
    parser.add_argument("--debug", default=False, action='store_true', help="如果引发异常，则删除调度器.")
    parser.add_argument('--representation-size', default=64, type=int, help='每个节点的向量表示维度.')
    parser.add_argument('--walk-length', default=40, type=int, help='每个节点随机游走的长度大小.')
    parser.add_argument('--window-size', default=5, type=int, help='skip-gram模型的窗口大小.')
    parser.add_argument('--number-walks', default=10, type=int, help='每个节点随机游走的次数')
    parser.add_argument('--undirected', default=True, type=bool, help='待处理的图是无向图.')
    parser.add_argument("-l", "--log", default="INFO", dest="log", help="日志冗长等级")
    parser.add_argument('--seed', default=0, type=int, help='随机游走生成器的seed.')
    parser.add_argument('--workers', default=1, type=int, help='并行处理的数量.')
    parser.add_argument('--format', default='adjlist', help='输入文件的文件格式.')

    parser.add_argument('--vertex-freq-degree', default=False, action='store_true', help='Use vertex degree to estimate the frequency of nodes in the random walks. This option is faster than calculating the vocabulary.')
    parser.add_argument('--matfile-variable-name', default='network', help='variable name of adjacency matrix inside a .mat file.')

    # 日志信息的配置
    args = parser.parse_args()
    numeric_level = getattr(logging, args.log.upper(), None)
    logging.basicConfig(format=LOGFORMAT)
    logger.setLevel(numeric_level)

    if args.debug:
        sys.excepthook = debug

    # 开始处理deepwalk
    process(args)


if __name__ == "__main__":

    # 第一种邻接表的输入
    input_path = "/Users/huangning/ByteCode/magellan_ai/magellan_ai/dl/mag_graph/deepwalk/example_graphs/karate.adjlist"
    output_path = "/Users/huangning/ByteCode/magellan_ai/magellan_ai/dl/mag_graph/deepwalk/example_graphs/karate" \
                  ".embedding "

    # 第一种edge表的输入
    input_path2 = "/Users/huangning/ByteCode/magellan_ai/magellan_ai/dl/mag_graph/deepwalk/example_graphs/blogcatalog" \
                  ".mat "
    output_path2 = "/Users/huangning/ByteCode/magellan_ai/magellan_ai/dl/mag_graph/deepwalk/example_graphs" \
                   "/blogcatalog.embedding "

    deepwalk_entry(input_path, output_path)
