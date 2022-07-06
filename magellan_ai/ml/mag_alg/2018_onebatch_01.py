"""
    寻找最大点集
"""

import datetime


# 方法1: 暴力搜素，时间复杂度O(n^2)
def violence_search(coordinate_list):
    start_time = datetime.datetime.now()
    max_cordi = []
    for cor in coordinate_list:
        count = 0
        for other in coordinate_list:
            if cor != other and cor[0] < other[0] and cor[1] < other[1]:
                count += 1
                break

        # 遍历一遍其他点之后，没有比当前点大的点，因此该点为"最大点"
        if count == 0:
            max_cordi.append((cor[0], cor[1]))
    end_time = datetime.datetime.now()
    time_len = end_time - start_time

    return max_cordi, time_len

# 方法2: 预排序 O(nlogn + n) = O(nlogn)
def presort_search(coordinate_list):
    start_time = datetime.datetime.now()
    max_cordi = []
    sorted_coordinate_list = sorted(coordinate_list, key=lambda x: (x[1], x[0]), reverse=True)
    print("排序后的列表为", sorted_coordinate_list)

    max_value =-1
    for elem in sorted_coordinate_list:
        if elem[0] > max_value:
            max_cordi.append(elem[1])
            max_value = elem[0]
    end_time = datetime.datetime.now()
    time_len = end_time - start_time

    return max_cordi, time_len


if __name__ == "__main__":

    print("请输入数据行数")
    n = int(input())

    coordinate = []
    for i in range(n):
        tmp = input().split(" ")
        coordinate.append((int(tmp[0]), int(tmp[1])))

    max_coordinate, time_length = violence_search(coordinate)
    max_coordinate2, time_length2 = presort_search(coordinate)

    print("法1-暴力法, 计算结果:{}:时间长度:{}".format(max_coordinate, time_length))
    print("法2-预排序法, 计算结果:{}:时间长度:{}".format(max_coordinate2, time_length2))