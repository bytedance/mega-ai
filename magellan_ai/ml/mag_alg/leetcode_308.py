"""
    题目: 最佳买卖股票时机含冷冻期(后续改变增加最多两次交易)
    给定一个整数数组prices，其中第prices[i]表示第i天的股票价格。
    设计一个算法计算出最大利润。在满足以下约束条件下，你可以尽可能地完成更多的交易（多次买卖一支股票）:
    卖出股票后，你无法在第二天买入股票 (即冷冻期为 1 天)。
    注意：你不能同时参与多笔交易（你必须在再次购买前出售掉之前的股票）。
"""


# 时间复杂度O(n), 空间复杂度O(n)
def max_profit(fee, prices):

    n = len(prices)
    dp = [[0]*2 for _ in range(n)]

    # 第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1
    for i in range(len(prices)):

        # base case 1
        if i == 0:
            dp[i][0] = 0
            dp[i][1] = -prices[i]
            continue

        # base case 2
        if i == 1:
            # 前一次买入，这次卖出 and 前一次没有买入
            dp[i][0] = max(dp[i-1][1]+prices[i], dp[i-1][0])

            # 前一次没买入，这次买入 and 前一次买入，这次不操作
            dp[i][1] = max(-prices[i], dp[i-1][1])

            continue

        dp[i][0] = max(dp[i-1][1] + prices[i], dp[i-1][0])

        # 下面这个情况最关键：上次没有买入，本次买入，并且保证度过冷冻期，and 上次就买入，本次不操作
        dp[i][1] = max(dp[i-2][0] - prices[i], dp[i-1][1])

    return dp[n-1][0]


if __name__ == "__main__":

    fee = 3
    prices=[3,3,5,0,0,3,1,4]

    dp = max_profit(fee, prices,2)
    print(dp)

