"""
    题目: 爱笑的奶茶店小姐姐(滑动窗口)
    有一个奶茶店小姐姐，她的奶茶店开了n分钟。每分钟都有一些顾客进入这家商店。给定一个长度为 n 的整数数组 customers ，其中 customers[i] 是在第 i 分钟开始时进入商店的顾客数量，
    所有这些顾客在第 i 分钟结束后离开。在某些时候，奶茶店小姐姐会笑。 如果奶茶店小姐姐在第 i 分钟笑，那么 smile[i] = 1，否则 smile[i] = 0。当奶茶店小姐姐微笑时，
    那一分钟的顾客就会满意，若奶茶店小姐姐不笑则顾客是不满意的。奶茶店小姐姐知道一个秘密技巧，能调高自己的情绪，可以让自己连续minutes分钟一直笑，但却只能使用一次。
    请你返回这一天营业下来，最多有多少客户能够感到满意。
"""


# 时间复杂度O(n), 秘密技巧只用来挽留, 将问题拆解成留住的的挽留的人群
def max_customer(customers, smile, minutes):

    # 所有不生气时间内的顾客总数 + 先计算起始的[0, X)分钟内挽留不满意的客户
    result = total_sum = sum([customers[i] for i in range(len(customers)) if smile[i]==1 or i < minutes])
    print(result)

    for i in range(minutes, len(customers)):
        total_sum = total_sum + customers[i] * (1-smile[i]) - customers[i-minutes] * (1-smile[i-minutes])
        result = max(result, total_sum)

    return result


if __name__ == "__main__":

    print("请输入顾客数组")
    customers = list(map(int, input().split(",")))
    print("请输入生气数组")
    grumpy = list(map(int, input().split(",")))
    print("奶茶店保持微笑")
    minutes = int(input())

    res = max_customer(customers, grumpy, minutes)
    print("最大满意的顾客数量为:", res)
