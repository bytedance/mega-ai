from sklearn.datasets import make_s_curve
from sklearn.manifold import Isomap
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np


"""
    Author: huangning
    Date: 2021/11/16
    Func: 指引工具
"""


def main():
    print("欢迎来到magellan_ai 👏")


    X, Y = make_s_curve(n_samples=500, noise=0.1, random_state=42)


    # 利用sklearn计算ISOMAP
    data_ISOMAP2 = Isomap(n_neighbors=10, n_components=2).fit_transform(X)
    plt.figure()
    plt.title("sk_Isomap")
    plt.scatter(data_ISOMAP2[:,0], data_ISOMAP2[:,1], c=Y)
    _ = plt.show(block=False)

if __name__ == "__main__":
    main()
