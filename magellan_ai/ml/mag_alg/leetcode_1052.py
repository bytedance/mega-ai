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


if __name__ == "__main__":

    minutes = int(input().split(",")[1])
    customers = list(map(int, input().split(",")))
    grumpy = list(map(int, input().split(",")))

    res = max_customer(customers, grumpy, minutes)
    print(res)