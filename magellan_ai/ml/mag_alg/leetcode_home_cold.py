"""
    题目: 最佳买卖股票时机含冷冻期(后续改变增加最多两次交易)
    给定一个整数数组prices，其中第prices[i]表示第i天的房屋价格；卖出房屋后，你无法在第二天买入房屋 (即冷冻期为 1 天)。你最多可以完成3笔交易。如果你已经购买了一个房屋，在卖出它之前你就不能再继续购买房屋了。返回获得利润的最大值。
    注意：这里的一笔交易指买入持有并卖出房屋的整个过程，每笔交易你只需要支付一次手续费。
"""


# 时间复杂度O(n), 空间复杂度O(n)
def max_profit(prices):

    max_buy_num = 3
    n = len(prices)
    dp = [[[0]*2 for _ in range(max_buy_num+1)] for _ in range(n)]

    # 第 i 天选择 buy 的时候，要从 i-2 的状态转移，而不是 i-1
    for i in range(len(prices)):
        for k in range(max_buy_num, 0, -1):
            # base case 1
            if i == 0:
                dp[i][k][0] = 0
                dp[i][k][1] = -prices[i]
                continue

            # base case 2
            if i == 1:
                # 前一次买入，这次卖出 and 前一次没有买入
                dp[i][k][0] = max(dp[i-1][k][1]+prices[i], dp[i-1][k][0])

                # 前一次没买入，这次买入 and 前一次买入，这次不操作
                dp[i][k][1] = max(-prices[i], dp[i-1][k][1])
                continue

            # 上次持有，本次卖出 and 上次未持有
            dp[i][k][0] = max(dp[i-1][k][1] + prices[i], dp[i-1][k][0])
            # 下面这个情况最关键：上次没有买入，本次买入，并且保证度过冷冻期，and 上次就买入，本次不操作
            dp[i][k][1] = max(dp[i-2][k-1][0] - prices[i], dp[i-1][k][1])

    return dp[n-1][max_buy_num][0]


if __name__ == "__main__":

    # fee = 3
    # prices=[3,3,5,0,0,3,1,4]

    prices = list(map(int, input().split(",")))

    dp = max_profit(prices)
    print(dp)

