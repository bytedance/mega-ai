from concurrent.futures import ProcessPoolExecutor
from multiprocessing import cpu_count
from collections import Counter
from time import time
from io import open
from os import path
import logging
import random
import graph

logger = logging.getLogger("deepwalk")

__current_graph = None

# speed up the string encoding
__vertex2str = None


def count_words(file):
    """ Counts the word frequences in a list of sentences.

  Note:
    This is a helper function for parallel execution of `Vocabulary.from_text`
    method.
  """
    c = Counter()
    with open(file, 'r') as f:

        for l in f:
            words = l.strip().split()
            c.update(words)
    return c


def count_textfiles(files, workers=1):
    c = Counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        for c_ in executor.map(count_words, files):
            c.update(c_)
    return c


def count_lines(f):
    if path.isfile(f):
        num_lines = sum(1 for line in open(f))
        return num_lines
    else:
        return 0


def _write_walks_to_disk(args):
    num_paths, path_length, alpha, rand, f = args
    G = __current_graph
    t_0 = time()
    with open(f, 'w') as fout:
        for walk in graph.build_deepwalk_corpus_iter(G=G, num_paths=num_paths, path_length=path_length,
                                                     alpha=alpha, rand=rand):
            fout.write(u"{}\n".format(u" ".join(v for v in walk)))
    logger.debug("Generated new file {}, it took {} seconds".format(f, time() - t_0))
    return f


def write_walks_to_disk(G, file_path, number_walks, path_length, alpha=0, rand=random.Random(0),
                        number_workers=cpu_count(), always_rebuild=True):
    global __current_graph
    __current_graph = G
    files_list = ["{}.{}".format(file_path, str(x)) for x in list(range(number_walks))]
    expected_size = len(G)
    args_list = []
    files = []

    # 若每个节点的随机游走的次数小于并行数量
    if number_walks <= number_workers:
        paths_per_worker = [1 for _ in range(number_walks)]
    else:
        # 针对并行的worker数量分配任务
        number_sub_walk = int(number_walks / number_workers) + 1
        paths_per_worker = [len(list(filter(lambda z: z is not None, [y for y in x]))) for x in
                            graph.grouper(number_sub_walk, range(1, number_walks + 1))]

    # ProcessPoolExecutor 用法
    with ProcessPoolExecutor(max_workers=number_workers) as executor:
        for size, file_, ppw in zip(executor.map(count_lines, files_list), files_list, paths_per_worker):
            if always_rebuild or size != (ppw * expected_size):
                args_list.append((ppw, path_length, alpha, random.Random(rand.randint(0, 2 ** 31)), file_))
            else:
                files.append(file_)

    with ProcessPoolExecutor(max_workers=number_workers) as executor:
        for file_ in executor.map(_write_walks_to_disk, args_list):
            files.append(file_)

    return files


class WalksCorpus(object):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for file in self.file_list:
            with open(file, 'r') as f:
                for line in f:
                    yield line.split()


def combine_files_iter(file_list):
    for file in file_list:
        with open(file, 'r') as f:
            for line in f:
                yield line.split()
