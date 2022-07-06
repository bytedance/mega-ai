"""
    题目: 爱生气的书店老板
    有一个书店老板，他的书店开了n分钟。每分钟都有一些顾客进入这家商店。给定一个长度为 n 的整数数组 customers ，其中 customers[i] 是在第 i 分钟开始时进入商店的顾客数量，所有这些顾客在第 i 分钟结束后离开
    在某些时候，书店老板会生气。 如果书店老板在第 i 分钟生气，那么 grumpy[i] = 1，否则 grumpy[i] = 0。
    当书店老板生气时，那一分钟的顾客就会不满意，若老板不生气则顾客是满意的。
    书店老板知道一个秘密技巧，能抑制自己的情绪，可以让自己连续minutes分钟不生气，但却只能使用一次。
    请你返回 这一天营业下来，最多有多少客户能够感到满意
"""


# 时间复杂度O(n*n), 超时
def max_customer(customers, grumpy, minutes):

    # 维持长度为3的窗口
    left, right = 0, minutes-1

    # 如果右指针大于等于列表最后一个位置索引
    if right >= len(grumpy)-1:
        return sum(customers)

    max_value = 0
    while right < len(grumpy):

        max_customer_num = 0
        max_customer_num += sum(customers[left: right+1])

        # 遍历滑动窗口左边的元素和
        for i in range(0, left):
            if grumpy[i] == 0:
                max_customer_num += customers[i]

        # 遍历滑动窗口右边的元素和
        for i in range(right+1, len(grumpy)):
            if grumpy[i] == 0:
                max_customer_num += customers[i]

        if max_customer_num > max_value:
            max_value = max_customer_num

        # 双指针同时加1
        left += 1
        right += 1

    return max_value

# 时间复杂度O(n), 秘密技巧只用来挽留, 将问题拆解成留住的的挽留的人群
def max_customer(customers, grumpy, minutes):

    # 所有不生气时间内的顾客总数 + 先计算起始的[0, X)分钟内挽留不满意的客户
    result = total_sum = sum([customers[i] for i in range(len(customers)) if grumpy[i]==0 or i < minutes])

    for i in range(minutes, len(customers)):
        total_sum = total_sum + customers[i] * grumpy[i] - customers[i-minutes] * grumpy[i-minutes]
        result = max(result, total_sum)

    return result


if __name__ == "__main__":

    print("请输入顾客数组")
    customers = list(map(int, input().split(",")))
    print("请输入生气数组")
    grumpy = list(map(int, input().split(",")))
    print("书店老板保持冷静")
    minutes = int(input())

    res = max_customer(customers, grumpy, minutes)
    print("最大满意的顾客数量为:", res)
