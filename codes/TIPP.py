import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def MaxDrawdown(return_list):
    '''计算最大回撤率'''
    i = np.argmax((np.maximum.accumulate(return_list) - return_list) /
                  np.maximum.accumulate(return_list))
    if i == 0:
        return 0
    j = np.argmax(return_list[:i])
    return ((return_list[j] - return_list[i]) / (return_list[j]))


def calc_rate(day, rate, rate_type):
    if rate_type == 0:
        return (1 + rate * day)
    elif rate_type == 1:
        return np.exp(rate * day)


def getData(filename):
    '''处理数据，获得各年的日涨跌幅'''
    raw_data = pd.read_excel(filename)[['日期', 'rate']]
    raw_data.rename(columns={'日期': 'day', 'rate': 'random_ret'}, inplace=True)
    raw_data['day'] = pd.to_datetime(raw_data['day'], format='%Y%m%d')
    raw_data.set_index('day', inplace=True)

    year = raw_data.resample('y')
    year.sum()  # 做一次无意义的运算，让year可以用来循环
    data = [j for i, j in year]
    return data


def getParameters(is_take_profit=0, test_num=1, rate_type=0, trading_year=1, trading_day_per_year=255, rf=0.04,
                  init_nav=1000000, adj_period=5, guarantee_rate=0.8, risk_multipler=2, risk_trading_fee_rate=0.006,
                  gaurant_adj_thresh=0.03, gaurant_inc=0.02, take_profit_thresh=0.12, gaurant_inc_counter=0):
    '''设置参数用来存储所需的固定金融参数初始化的数值'''
    parameters = {'is_take_profit': is_take_profit, 'test_num': test_num, 'rate_type': rate_type,
                  'trading_year': trading_year, 'rf': rf, 'trading_day_per_year': trading_day_per_year,
                  'rf_daily': rf / trading_day_per_year, 'trading_day_sum': trading_year * trading_day_per_year,
                  'init_nav': init_nav, 'adj_period': adj_period, 'guarantee_rate': guarantee_rate,
                  'risk_multipler': risk_multipler, 'risk_trading_fee_rate': risk_trading_fee_rate,
                  'gaurant_adj_thresh': gaurant_adj_thresh, 'gaurant_inc': gaurant_inc,
                  'take_profit_thresh': take_profit_thresh, 'gaurant_inc_counter': gaurant_inc_counter}

    return parameters


def outputQuantResult(Return, nav, trading_day_sum):
    '''输入经过该策略后的时间序列结果'''
    annual_return, annual_volatility, Sharpe, Maxdrawdown = [], [], [], []
    for i in range(len(Return)):
        df_return = pd.DataFrame(Return[i])
        annual_return.append(Return[i][len(Return[i]) - 1])
        volatility = (df_return.shift(1) - df_return) / df_return
        annual_volatility.append(
            float(volatility.std() * np.sqrt(trading_day_sum)))
        Sharpe.append((annual_return[i] - 1) / annual_volatility[i])
        Maxdrawdown.append(float(df_return.apply(MaxDrawdown, axis=0)))

    Results = pd.DataFrame()
    Results['annual_return'] = annual_return
    Results['annual_volatility'] = annual_volatility
    Results['Sharpe'] = Sharpe
    Results['Maxdrawdown'] = Maxdrawdown

    # 绘制每期收益图
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.xlabel('时间(天)')
    plt.ylabel('净值(元)')
    plt.plot(range(1, len(nav)), nav[1:])

    return Results


if __name__ == "__main__":
    data = getData(filename='600519_SH.xlsx')
    Return = []
    for i in range(len(data)):
        p = getParameters(trading_day_per_year=data[i].shape[0])

        risk_asset = np.zeros(p["trading_day_sum"])  # 风险资产
        rf_asset = np.zeros(p["trading_day_sum"])  # 无风险资产
        min_pv_asset = np.zeros(p["trading_day_sum"])  # 价值底线
        nav = np.zeros(p["trading_day_sum"])  # 总资产
        nav[1] = p["init_nav"]

        # TIPP策略
        # 第1天
        min_pv_asset[1] = p["guarantee_rate"] * p["init_nav"] / \
                          calc_rate(p["trading_day_sum"], p["rf_daily"], p["rate_type"])  # 第1天的价值底线
        risk_asset[1] = max(0, p["risk_multipler"] * (nav[1] -
                                                      min_pv_asset[1]))  # 风险资产 w/o fee
        rf_asset[1] = (nav[1] - risk_asset[1])  # 无风险资产
        risk_asset[1] = risk_asset[1] * (1 - p["risk_trading_fee_rate"])  # 扣除手续费
        # 第2天到最后1天
        for t in range(2, p["trading_day_sum"]):
            # 止盈情况
            if p["is_take_profit"] == 0:
                # 检查是否已经可以止盈
                fv = nav[t - 1] * calc_rate(p["trading_day_sum"] - t, p["rf_daily"], p["rate_type"]) - \
                     p["risk_trading_fee_rate"] * risk_asset[t - 1]  # 检查减去所有的手续费后，是否还满足止盈条件
                if fv / p["init_nav"] - 1 > p["take_profit_thresh"]:  # 止盈的情况
                    risk_asset[t] = 0
                    rf_asset[t] = rf_asset[t - 1] * calc_rate(1, p["rf_daily"], p["rate_type"]) + (
                            1 - p["risk_trading_fee_rate"]) * risk_asset[t - 1]
                    p["is_take_profit"] = 1
                    nav[t] = rf_asset[t]
                else:  # 没有止盈的情况
                    # 如果已实现收益，提高保本额度
                    if nav[t - 1] / p["init_nav"] > p["guarantee_rate"] * (p["gaurant_adj_thresh"] + 1):
                        p["guarantee_rate"] += p["gaurant_inc"]
                        p["gaurant_inc_counter"] += 1

                    min_pv_asset[t] = p["guarantee_rate"] * p["init_nav"] / \
                                      calc_rate(p["trading_day_sum"] - t + 1,
                                                p["rf_daily"], p["rate_type"])  # 价值底线
                    risk_asset[t] = (1 + data[i].iloc[t - 1]) * risk_asset[t - 1]
                    # risk_asset[t] = (1 + random_ret[0][t-1] ) * risk_asset[t -1]
                    rf_asset[t] = calc_rate(
                        1, p["rf_daily"], p["rate_type"]) * rf_asset[t - 1]
                    nav[t] = risk_asset[t] + rf_asset[t]

                    # 进行定期调整
                    if (t - 1) % p["adj_period"] == 0:
                        risk_asset_b4_adj = risk_asset[t]
                        risk_asset[t] = max(
                            0, p["risk_multipler"] * (nav[t] - min_pv_asset[t]))  # 风险资产
                        rf_asset[t] = nav[t] - risk_asset[t]  # 无风险资产
                        trade_value = risk_asset_b4_adj - risk_asset[t]
                        risk_asset[t] = risk_asset[t] - \
                                        abs(trade_value) * p["risk_trading_fee_rate"]  # 手续费

                    # 检查是否被强制平仓
                    if risk_asset[t] <= 0:
                        rf_asset[t] = nav[t] - risk_asset[t] * \
                                      p["risk_trading_fee_rate"]
                        risk_asset[t] = 0
            else:
                # 止盈
                rf_asset[t] = rf_asset[t - 1] * \
                              calc_rate(1, p["rf_daily"], p["rate_type"])
                nav[t] = rf_asset[t]
        Return.append(nav[1:] / p["init_nav"])

    Results = outputQuantResult(Return, nav, p["trading_day_sum"])
    Results.to_excel("results.xlsx")



