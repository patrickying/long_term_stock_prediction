from yahoofinancials import YahooFinancials
import numpy as np
import wbdata
import pandas as pd
import math
from datetime import datetime
import requests
import json


def add_month(ts, m):
    ts_adj = datetime(ts.year + (ts.month + m // 12), ((ts.month + m % 12)), 1).strftime("%Y-%m-%d")
    return ts_adj

def get_financialstatement_data():
    ticker = ['AAPL']
    indicator = ['Revenue', 'Gross Profit', 'R&D Expenses', 'Net Income', 'EBITDA Margin']
    s = "https://financialmodelingprep.com/api/v3/financials/income-statement/%s?period=quarter" % ticker[0]
    r = requests.get(s)
    # print(type(dic))
    json_dict = json.loads(r.text)
    fs = json_dict['financials']
    X = []
    for x in fs:
        temp = [x['date']]
        for i in indicator:
            temp.append(float(x[i]))
        X.append(temp)
    X = sorted(X)
    x_yoy = []
    for i in range(4, len(X)):
        temp = [X[i][0]]
        for i2 in range(1, len(X[0])):
            temp.append(X[i][i2] / X[i - 4][i2])
        x_yoy.append(temp)
    return x_yoy


def get_worldbank_data():
    # wbdata.search_countries("united")
    countries = ['USA']
    indicators = {"NY.GDP.PCAP.PP.KD": "gdppc"}
    df = wbdata.get_dataframe(indicators, country=countries, convert_date=True)
    dic = df.to_dict()['gdppc']
    X = sorted([[add_month(k, 15), v] for k, v in dic.items() if not (math.isnan(v))])
    x_yoy = []
    for i in range(1, len(X)):
        x_yoy.append([X[i][0], X[i][1] / X[i - 1][1]])
    return x_yoy


def get_stock_price_day():
    ticker = ['AAPL']
    date_list = []
    yahoo_financials = YahooFinancials('AAPL')
    historical_stock_prices = yahoo_financials.get_historical_price_data('2000-07-01', '2019-09-01', 'daily')
    for data in historical_stock_prices['AAPL']['prices']:
        date_list.append(data['formatted_date'])
    date_list = sorted(date_list)

    X = []
    for index_t, t in enumerate(ticker):
        yahoo_financials = YahooFinancials(ticker)
        historical_stock_prices = yahoo_financials.get_historical_price_data('2000-07-01', '2019-09-01', 'daily')

        close = [0] * len(date_list)
        high = [0] * len(date_list)
        low = [0] * len(date_list)
        for data in historical_stock_prices[t]['prices']:
            close[date_list.index(data['formatted_date'])] = data['close']

        # make sure there's no missing data
        for i in range(1, len(close)):
            if close[i] == 0 or close[i - 1] == 0:
                X.append([date_list[i], float('inf')])
            else:
                X.append([date_list[i], close[i] / close[i - 1]])

        cur_month = '08'
        CNN_X = []
        temp_X = []
        for x, y in X:
            temp_X.append(y)
            if len(temp_X) > 20:
                temp_X = temp_X[1:]
            if cur_month != x.split('-')[1]:
                cur_month = x.split('-')[1]
                CNN_X.append(["-".join(x.split('-')[0:2]) + "-01"] + temp_X)
    return CNN_X


def get_stock_price():
    ticker = ['AAPL']
    date_list = []
    yahoo_financials = YahooFinancials('AAPL')
    historical_stock_prices = yahoo_financials.get_historical_price_data('2000-09-01', '2019-09-01', 'monthly')
    for data in historical_stock_prices['AAPL']['prices']:
        date_list.append(data['formatted_date'])
    date_list = sorted(date_list)

    X = []
    for index_t, t in enumerate(ticker):
        yahoo_financials = YahooFinancials(ticker)
        historical_stock_prices = yahoo_financials.get_historical_price_data('2000-09-01', '2019-09-01', 'monthly')

        close = [0] * len(date_list)
        high = [0] * len(date_list)
        low = [0] * len(date_list)
        for data in historical_stock_prices[t]['prices']:
            close[date_list.index(data['formatted_date'])] = data['close']
            high[date_list.index(data['formatted_date'])] = data['high']
            low[date_list.index(data['formatted_date'])] = data['low']

        # make sure there's no missing data
        for i in range(1, len(close)):
            if close[i] == 0 or close[i - 1] == 0:
                X.append([date_list[i],float('inf'), float('inf'), float('inf')])
            else:
                X.append([date_list[i],close[i] / close[i - 1], high[i] / high[i - 1], low[i] / low[i - 1]])

    return X


def GRUD_data(raw, macro, mask_open=True):
    raw_dim = len(raw[0]) - 1  # the first column is date
    # combine raw, macro by date
    count = 0
    while raw[0][0] >= macro[count+1][0]:
        count += 1
    count = max(0, count - 1)
    for i in range(len(raw)):
        if raw[i][0] >= macro[count+1][0]:
            count += 1
            raw[i] += macro[count][1:]
        else:
            raw[i] += [float('inf')] * (len(macro[0]) - 1)

    # add mask won't del date when adding mask
    adj = [[x for x in raw[0]]]
    if mask_open:
        adj[0] += [1] * (len(raw[0]) - 1) * 2
    interval = [1] * (len(raw[0]) - 1)
    for i in range(1, len(raw)):
        temp = []
        mask = []
        for i1, d in enumerate(raw[i]):
            if i1 == 0: # date
                temp.append(d)
                continue
#             i2 = i1 - 1
            if d == float('inf'):
                temp.append(adj[-1][i1])
                mask.append(0)
                interval[i1-1] += 1
            else:
                temp.append(d)
                mask.append(1)
                interval[i1-1] = 1
        if mask_open:
            adj.append(temp + mask + interval)
        else:
            adj.append(temp)

    return adj


def to_txt(data, name):
    with open(name, 'w') as f:
        for x in data:
            for i in range(len(x)):
                # f.write("{0:.4f}".format(x[i]))
                f.write(str(x[i]))
                if i != len(x)-1:
                    f.write(',')
            f.write('\n')

if __name__ == '__main__':
    raw_data = get_stock_price()
    macro_data = get_worldbank_data()
    stock_day = get_stock_price_day()
    adj_data = GRUD_data(raw_data, macro_data)
    adj_data = GRUD_data(adj_data, stock_day, False)
    to_txt(adj_data, 'stock.txt')