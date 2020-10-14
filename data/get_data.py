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

#显示所有列
pd.set_option('display.max_columns', None)
#显示所有行
pd.set_option('display.max_rows', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

ts.set_token('ae0addf484ab2b76fe78cdb46b10165b48d31748a48202f9df193951')
pro = ts.pro_api()
start_date = '19990101'
end_date = '20200601'


# 获取股价
def get_stock_price(ticker):
    df = pro.daily(ts_code=ticker, start_date=start_date, end_date=end_date)
    df.to_csv('price_chinese/' + str(ticker) + '.csv',index=0)
    return df


# # 财报数据
def get_financialstatement_data(ticker):
    # income
    income_indicator = ['end_date','ann_date','f_ann_date', 'total_revenue', 'total_cogs', 'int_exp', 'oper_exp', 'n_income','ebit','ebitda','basic_eps','operate_profit']
    ds1 = pro.income(ts_code=ticker, start_date=start_date, end_date=end_date, fields=','.join(income_indicator))
    ds1 = ds1.groupby('end_date').max()
    ds1.sort_values(by='end_date', inplace=True, ascending=False)
    # balance
    balance_indicator = ['end_date','accounts_receiv','acct_payable','inventories','amor_exp','total_cur_assets','intan_assets',                        'r_and_d','goodwill','total_assets','total_cur_liab','total_liab']
    ds2 = pro.balancesheet(ts_code=ticker, start_date=start_date, end_date=end_date, fields=','.join(balance_indicator))
    ds2 = ds2.groupby('end_date').max()
    ds2.sort_values(by='end_date', inplace=True, ascending=False)
    #cash
    ds3 = pro.cashflow(ts_code=ticker, start_date=start_date, end_date=end_date, fields='end_date,c_cash_equ_end_period')
    ds3 = ds3.groupby('end_date').max()
    ds3.sort_values(by='end_date', inplace=True, ascending=False)

    df = pd.merge(ds1.drop_duplicates(),ds2.drop_duplicates(),how='left',left_on=['end_date'],right_on=['end_date'])
    df = pd.merge(df.drop_duplicates(),ds3.drop_duplicates(),how='left',left_on=['end_date'],right_on=['end_date'])
    df.to_csv('fsa_chinese/' + str(ticker) + '.csv')
    return df


# 删除多的股票
def delete(folder):
    stock_name = set()
    for root, dirs, files in os.walk('train'):
        for f in files:
            if '.SH' in f:
                stock_name.add(f[:-4])

    for root, dirs, files in os.walk(folder):
        for f in files:
            if f[:-4] not in stock_name:
                os.remove(folder + '/' + f)

# 检查数据缺失率
def fsa_check():
    for root, dirs, files in os.walk('fsa_chinese'):
        df_total = pd.read_csv('fsa_chinese/' + files[0])
        df_total = df_total.loc[df_total['end_date']>=20100101]
        for f in files[1:]:
            df = pd.read_csv('fsa_chinese/' + f)
            df = df.loc[df['end_date']>=20100101]
            df_total = pd.concat([df_total, df])
    df_total = df_total.replace(0, np.nan)
    print(df_total.count())
    print(df_total.isnull().sum(axis=0).tolist())
    # >= 70% ['basic_eps','total_revenue','total_cogs','operate_profit','n_income','ebit','accounts_receiv,'inventories',\
    # 'total_cur_assets','intan_assets','total_assets','acct_payable','total_cur_liab','total_liab','c_cash_equ_end_period']



if __name__ == '__main__':
    # get stock fs
    # data = pro.stock_basic(exchange='', list_status='L', fields='ts_code,list_date')
    # data = data.values.tolist()
    # count = 0
    # for x in data:
    #     if x[1] <= '20100101':
    #         get_financialstatement_data(x[0])
    #         time.sleep(1)  # 50/sec
    #     count += 1
    #     if count % 100 == 0:
    #         print(count,len(data))

    #delete
    # delete('fsa_chinese')
    # fsa_check()

    get_financialstatement_data('600589.SH')
