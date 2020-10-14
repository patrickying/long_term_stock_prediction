# coding: utf-8
import copy
import tushare as ts
import numpy as np
import wbdata
import pandas as pd
import math
from datetime import datetime, timedelta
import requests
import json
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
from datetime import datetime
from dateutil.relativedelta import relativedelta
import time
import os

ts.set_token('ae0addf484ab2b76fe78cdb46b10165b48d31748a48202f9df193951')
pro = ts.pro_api()
start_date = '20000101'
end_date = '20200601'


def add_month(ts, m):
    ts_adj = datetime(ts.year + (ts.month + m // 12), ((ts.month + m % 12)), 1).strftime("%Y-%m-%d")
    return ts_adj


def to_txt(data, name):
    temp = 'w'
#     if ticker == 'AAPL':temp = 'w'
    with open(name, temp) as f:
        for i2, x in enumerate(data[:-1]):
            for i in range(1,len(x)):
                f.write(str(x[i]))
                f.write(',')
                if i == len(x)-1:
                    f.write(str(data[i2+1][1]))
                    
            f.write('\n')


# # 财报数据
# def get_financialstatement_data(ticker):
#     # income
#     income_indicator = ['end_date', 'total_revenue', 'total_cogs', 'int_exp', 'oper_exp', 'n_income','ebit','ebitda','basic_eps','operate_profit']
#     ds1 = pro.income(ts_code=ticker, start_date=start_date, end_date=end_date, fields=','.join(income_indicator))
#     # balance
#     balance_indicator = ['end_date','accounts_receiv','acct_payable','inventories','amor_exp','total_cur_assets','intan_assets',                        'r_and_d','goodwill','total_assets','total_cur_liab','total_liab']
#     ds2 = pro.balancesheet(ts_code=ticker, start_date=start_date, end_date=end_date, fields=','.join(balance_indicator))
#     #cash
#     ds3 = pro.cashflow(ts_code=ticker, start_date=start_date, end_date=end_date, fields='end_date,c_cash_equ_end_period')
#     df = pd.merge(ds1.drop_duplicates(),ds2.drop_duplicates(),how='left',left_on=['end_date'],right_on=['end_date'])
#     df = pd.merge(df.drop_duplicates(),ds3.drop_duplicates(),how='left',left_on=['end_date'],right_on=['end_date'])
#     df.to_csv('fsa_chinese/' + str(ticker) + '.csv',index=0)
#     return df


# Stock Price
def get_stock_price(ticker):
    df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
    df.to_csv('price_chinese/' + str(ticker) + '.csv',index=0)
    return df


def delta(date, data):
    # 获取对应日期的数据，补缺失值
    dic = {}
    dic_data = dict()
    i = 0
    temp = data[0][1:]
    # 补缺失值(前值)
    for x in date:
        while x >= data[i][0]:
            temp = data[i][1:]
            i += 1
        i -= 1
        dic_data[x] = temp
    
    # raw->diff
    result_list = []
    for x in range(1,len(date)):
        cur = date[x]
        pre = date[x-1]
        dic[cur] = list(map(lambda x: min(max(round(x[0]/x[1], 6), 0.9), 1.1), zip(dic_data[cur], dic_data[pre])))
        if x != 1: dic[pre].append(dic[cur][-1])  # y label

    # raw->ta
    # dic = technical_analysis(dic, dic_data, date)



    return dic


# 每一家公司计算财务比率后，再计算相较于市场的Z值
def fsa_ratio():
    # 缺失率<30%的指标
    indicator = ['basic_eps','total_revenue','total_cogs','operate_profit','n_income','ebit','accounts_receiv','inventories',\
        'total_cur_assets','intan_assets','total_assets','acct_payable','total_cur_liab','total_liab','c_cash_equ_end_period']
    indicator_ratio = []
    # 读取全部财报的均值跟std
    stock = {}
    for root, dirs, files in os.walk('fsa_chinese'):
        df_total = pd.read_csv('fsa_chinese/' + files[0])
        df_total = df_total.loc[df_total['end_date'] >= 20090101]
        df_total = df_total.replace(0, np.nan)
        df_total = df_total.set_index('end_date')
        stock[files[0][:-4]] = df_total
        for f in files[1:]:
            df = pd.read_csv('fsa_chinese/' + f)
            df = df.loc[df['end_date'] >= 20090101]
            df = df.replace(0, np.nan) # 避免计算错误
            df = df.set_index('end_date')
            stock[f[:-4]] = df
            df_total = pd.concat([df_total, df],sort=False)

    for i in range(1, len(indicator)):
        for i2 in range(i, len(indicator)):
            if i != i2:
                ind1, ind2 = indicator[i], indicator[i2]
                df_total[ind1 + '/' + ind2] = df_total[ind1] / df_total[ind2]
                indicator_ratio.append(ind1 + '/' + ind2)
    df_std = df_total.groupby('end_date')[indicator+indicator_ratio].std()
    df_avg = df_total.groupby('end_date')[indicator+indicator_ratio].mean()

    for k, v in stock.items():
        date = v.index.tolist()
        # eps
        ind1 = indicator[0]
        stock[k][ind1 + '_Z'] = v[ind1]
        for d in date:
            stock[k].at[d, ind1 + '_Z'] -= df_avg.at[d, ind1]
            stock[k].at[d, ind1 + '_Z'] /= df_std.at[d, ind1]
            stock[k].at[d, ind1 + '_Z'] = \
                min(max(stock[k].at[d, ind1], -3), 3)
        for i in range(1, len(indicator)):
            for i2 in range(i, len(indicator)):
                if i != i2:
                    ind1, ind2 = indicator[i], indicator[i2]
                    stock[k][ind1 + '/' + ind2] = v[ind1] / v[ind2]
                    for d in date:
                        stock[k].at[d, ind1+'/'+ind2] -= df_avg.at[d, ind1+'/'+ind2]
                        stock[k].at[d, ind1 + '/' + ind2] /= df_std.at[d, ind1+'/'+ind2]
                        stock[k].at[d, ind1 + '/' + ind2] = \
                            min(max(stock[k].at[d, ind1 + '/' + ind2], -3), 3)
        stock[k] = stock[k].reset_index()
        stock[k].to_csv('fsa_chinese_Z/' + str(k) + '.csv', index=0)


def fs_analysis(date, ticker):
    dic = {}
    df = pd.read_csv('fsa_chinese_Z/' + ticker + '.csv')
    df = df.replace(np.nan,0.0)
    indicator = [0] + [24 + x for x in range(92)]
    df = df.iloc[:, indicator]
    fsa_list = sorted([tuple(x) for x in df.values])
    count = 2
    for d in date:
        while count<len(fsa_list) and d >= str(int(fsa_list[count][0])):
            count += 1
        dic[d] = list(fsa_list[count-2][1:])

    return dic
# d = fs_analysis(['20110301','20110311','20110321','20110331','20110401','20110421'],'600004.SH')
# for k,v in d.items():
#     print(k,v)

def technical_analysis(dic, dic_data, date):  # dic:处理后 dic_data:原始数据, date:工作日
    # RSI https://zh.wikipedia.org/wiki/%E7%9B%B8%E5%B0%8D%E5%BC%B7%E5%BC%B1%E6%8C%87%E6%95%B8
    duration = 14
    U = 0.0
    D = 0.0
    for i in range(1, len(date)):
        cur = date[i]
        pre = date[i-1]
        cur_delta = dic_data[cur][2]-dic_data[pre][2]
        U = (1-(1/duration))*U + (1/duration)*max(cur_delta, 0)
        D = (1 - (1 / duration)) * D + (1 / duration) * max(-cur_delta, 0)
        if U+D == 0: rsi = 0
        else: rsi = U*100/(U+D)
        dic[cur] = [rsi] + dic[cur]

    # SMA 当前价格是否超过MA 1是超过
    duration = 10  # MA周期
    past_data = [dic_data[date[0]][2]]
    for i in range(1, len(date)):
        cur = date[i]
        price = dic_data[cur][2]
        past_data.append(price)
        if len(past_data) > duration:
            past_data = past_data[1:]
        if sum(past_data) / duration <= price:
            ma = 1
        else: ma = 0
        dic[cur] = [ma] + dic[cur]
    return dic


def output(folder):
    # stock list
    stock_dic = {}
    for root, dirs, files in os.walk('train_week'):
        for f in files:
            if '.SH' in f:
                stock_dic[f[:-4]] = []
    # trading day
    df = pro.trade_cal(exchange='', start_date='20100101', end_date='20191231')
    week = set()
    df = df.loc[df['is_open']==1]
    day = df['cal_date'].values.tolist()
    start = datetime.strptime('20100108','%Y%m%d')
    i = 0
    temp = ''
    while start.strftime("%Y%m%d") <= day[-1]:
        b = start.strftime("%Y%m%d")
        if day[i] > b:
            week.add(temp)
            start += timedelta(days=7)
        temp = day[i]
        i += 1
    week = sorted(list(week))
    # result by day
    result = dict(zip(day[1:],[[] for _ in range(len(day[1:]))]))

    # stock price
    for k in stock_dic.keys():
    # for k in ['600136.SH']:
        df = pd.read_csv('price_chinese/' + k + '.csv')
        price = df.loc[:,['trade_date','high','low','close']].values.tolist()
        for i in range(len(price)):
            price[i][0] = str(int(price[i][0]))
        price = sorted(price)
        if price[-1][0]<'20191231':
            continue
        dic = delta(week,price)
        # check = sum([1 for v in r if v == [1.0,1.0,1.0,1.0]])/len(day)
        # if check >= 0.05:  # 确保缺失数据不超过5%
        #     continue

        # add fsa
        fsa_dic = fs_analysis(week, k)

        # dict->list
        r = []
        for x in range(1,len(week)-1):  # 最后一个没有y值
            dic[week[x]] = fsa_dic[week[x]] + dic[week[x]]
            r.append(dic[week[x]])

        # to txt
        with open(folder + '/' + k + '.txt', 'w') as f:
            # for x in day[1:]:
                for data in r:
                    f.write(','.join(list(map(str,data))))
                    f.write('\n')

if __name__ == '__main__':
    # get stock price
    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date')
    # data = data.values.tolist()
    # count = 0
    # for x in data:
    #     if x[1] <= '20100101':
    #         get_financialstatement_data(x[0])
    #         time.sleep(1)  # 50/sec
    #         count += 1
    #     if count % 100 == 0: print(count)

    # output('train_week_fa')
    # delete('train_week')
    # fsa_ratio()
    pass

