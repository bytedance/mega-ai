"""
    题目: 买卖股票的最佳时机含手续费
    给定一个整数数组 prices，其中 prices[i]表示第 i 天的房屋价格 ；整数 fee 代表了交易房屋的手续费用。你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个房屋，在卖出它之前你就不能再继续购买房屋了。返回获得利润的最大值。你最多可以完成3笔交易
    注意：这里的一笔交易指买入持有并卖出房屋的整个过程，每笔交易你只需要为支付一次手续费。
"""


# 时间复杂度O(n), 空间复杂度O(n)
def max_profit(fee, prices):

    max_buy_num = 3
    n = len(prices)
    dp = [[[0]*2 for _ in range(max_buy_num+1)] for _ in range(n)]

    for i in range(len(prices)):
        for k in range(max_buy_num, 0, -1):
            if i == 0:
                dp[i][k][0] = 0
                dp[i][k][1] = -prices[i] - fee
                continue

            dp[i][k][0] = max(dp[i-1][k][1] + prices[i], dp[i-1][k][0])
            dp[i][k][1] = max(dp[i-1][k-1][0] - prices[i] - fee, dp[i-1][k][1])

    return dp[n-1][max_buy_num][0]


if __name__ == "__main__":

    fee = int(input().split(",")[1])
    prices = list(map(int, input().split(",")))

    # prices=[3,3,5,0,0,3,1,4]
    # prices = [1, 2, 3, 4, 5]
    # prices = [7,6,4,3,1]
    # prices = [1]

    dp = max_profit(fee, prices)
    print(dp)

