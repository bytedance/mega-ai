"""
    题目: 买卖股票的最佳时机含手续费
    给定一个整数数组prices，其中 prices[i]表示第i天的股票价格, 整数fee代表了交易股票的手续费用。
    你可以无限次地完成交易，但是你每笔交易都需要付手续费。如果你已经购买了一个股票，
    在卖出它之前你就不能再继续购买股票了。返回获得利润的最大值。
"""


# 时间复杂度O(n), 空间复杂度O(n)
def max_profit(fee, prices):
    n = len(prices)
    dp = [[0]*2 for _ in range(n)]

    dp_i_0 = -1
    dp_i_1 = -1

    # 枚举所有状态
    for i in range(len(prices)):
        if i == 0:
            # dp[i][0] = 0
            # dp[i][1] = -prices[i] - fee
            dp_i_0 = 0
            dp_i_1 = -prices[i] - fee
        else:
            tmp = dp_i_0
            dp_i_0 = max(dp_i_1 + prices[i], dp_i_0)
            dp_i_1 = max(tmp - prices[i] - fee, dp_i_1)

            # dp[i][0] = max(dp[i-1][1] + prices[i], dp[i-1][0])
            # dp[i][1] = max(dp[i-1][0] - prices[i] - fee, dp[i-1][1])

    return dp_i_0


if __name__ == "__main__":

    fee = int(input())
    prices = list(map(int, input().split(",")))

    dp = max_profit(fee, prices)
    print(dp)

