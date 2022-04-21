import os
import numpy as np
import pandas as pd
import pandas_datareader
import pandas_datareader.moex as moex
from lxml import etree, html
from lxml.etree import tostring
import re
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

sber_date_format = '%d.%m.%Y'

h1 = {
    'Торговая площадка': 'section',
    'Оценка портфеля ЦБ, руб.': 'cost',
    'Денежные средства, руб.': 'cash',
    'Оценка, руб.': 'all_cost',
    'Дата': 'date',
    'Описание операции': 'operation',
    'Валюта': 'currency',
    'Сумма зачисления': 'inflow',
    'Сумма списания': 'outflow',
    'Сумма': 'cash',
    'Наименование': 'shortname',
    'ISIN ценной бумаги': 'ISIN',
    'Валюта рыночной цены': 'currency',
    'Количество, шт': 'count',
    'Номинал*': 'value',
    'Рыночная цена **': 'market_price',
    'Рыночная стоимость, без НКД***': 'market_price2',
    'НКД****':'interest',
    'Вид, Категория, Тип, иная информация': 'sectype'}

# Загрузка таблицы оценки активов
def asset_valuation(t, contract_name, sd, ed):
    #t.remove(t[-1])
    df = pd.read_html(tostring(t), header=0)[0]

    df.rename(columns=h1, inplace=True)
    for h in ['cost', 'cash', 'all_cost']:
        if (df[h].dtype != np.float64):
            df[h] = pd.to_numeric(df[h].str.replace(' ', ''))

    # Проверка правильности загруженых значений
    asset_sum = (df['cost'][:-1]+df['cash'][:-1]).sum()
    if abs(asset_sum-df.iloc[-1,-1]) >= 1:
        raise Exception("Ошибка загрузки таблицы оценки активов")
    #res = df.iloc[:-1,:-1]
    #print(df)
    df = df.iloc[:-1,:-1]
    df['contract'] = contract_name
    df['date'] = ed
    df.set_index(['date', 'contract', 'section'], inplace=True)
    return df


def ind_inv_account(t, cc, sd, ed):
    return None

# ВНИМАНИЕ: данная функция работает только для площадки фондовый рынок
def securities_portfolio(t, cc, sd, ed):
    t.remove(t[-1])
    df = pd.read_html(tostring(t), header=0, skiprows=[0,2])[0]
    df.rename(columns=h1, inplace=True)
    df_1 = df.iloc[:,[0,1,2,3,4,5,6,7]].copy()
    df_2 = df.iloc[:,[0,1,2,8,9,10,11,12]].copy()
    df_2.columns=df_1.columns

    df_1['date'] = sd
    df_2['date'] = ed
    df = df_1.append(df_2)
    df['contract'] = cc
    df.set_index(['date', 'contract', 'ISIN'], inplace=True)
    #df['count'] = df['count'].astype(str)
    for h in ['count','value', 'market_price', 'market_price2', 'interest']:
        #if (np.issubdtype(df[h].dtype , np.number) is not True):
        df[h] = df[h].astype(str)
        df[h] = pd.to_numeric(df[h].str.replace(' ', ''))
    df = df[df['count']>0]
    return df


def cash(t, cc, sd, ed):
    return None

# Таблица движения денежных средств
def flow_of_funds(t, cc, sd, ed):
    #t.remove(t[-1])
    df = pd.read_html(tostring(t), header=0)[0]
    #df = df[:-1]
    df.rename(columns=h1, inplace=True)

    for h in ['inflow', 'outflow']:
        if (df[h].dtype != np.float64):
            df[h] = pd.to_numeric(df[h].str.replace(' ', ''))

        # Проверка правильности загруженых значений
        if abs(df[h][:-1].sum()-df[h][-1:].values[0]) > 1:
            raise Exception("Ошибка загрузки таблицы движения денежных средств")

    df = df[:-1]
    df['contract'] = cc
    df['date'] = pd.to_datetime(df['date'], format=sber_date_format)
    df.set_index(['date', 'contract', 'section'], inplace=True)
    return df


def deals(t, cc, sd, ed):
    return None


def repos(t, cc, sd, ed):
    return None


def payouts(t, cc, sd, ed):
    df = pd.read_html(tostring(t), header=0)[0]
    df.rename(columns=h1, inplace=True)
    df['date'] = pd.to_datetime(df['date'], format=sber_date_format)
    if (df['cash'].dtype != np.float64):
        df['cash'] = df['cash'].str.replace(' ', '')
        df['cash'] = pd.to_numeric(df['cash'])
    df['contract'] = cc
    #df.set_index(['date', 'contract', 'section'], inplace=True)
    return df


def securities_directory(t, cc, sd, ed):
    df = pd.read_html(tostring(t), header=0)[0]
    df.rename(columns=h1, inplace=True)
    df.set_index('ISIN', inplace=True)
    return df


table_names = {
    'Оценка активов': asset_valuation,
    'Информация о зачислениях денежных средств на ИИС': ind_inv_account,
    'Портфель Ценных Бумаг': securities_portfolio,
    'Денежные средства': cash,
    'Движение денежных средств за период': flow_of_funds,
    'Сделки купли/продажи ценных бумаг': deals,
    'Сделки РЕПО': repos,
    'Выплаты дохода от эмитента на внешний счет': payouts,
    'Справочник Ценных Бумаг**': securities_directory}


def load_reports(file_paths):
    tbls = {}

    date_pattern = re.compile(r'за период с ([0-9.]+) по ([0-9.]+), дата создания ([0-9.]+)')
    contract_pattern = re.compile(' [0-9A-Z]{5,10} ')

    for file_path in file_paths:
        #print(f'Файл {file_path}')
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as report_file:
            htmlstr = report_file.read()

        root = html.fromstring(htmlstr)
        test = root.xpath('/html/body/h3')

        string = test[0].text_content()
        match = date_pattern.search(string)
        start_date = dt.datetime.strptime(match[1], sber_date_format)
        #start_date = start_date.replace(hour=9, minute=55)
        end_date = dt.datetime.strptime(match[2], sber_date_format)
        #end_date= end_date.replace(hour=18, minute=55)
        report_date = dt.datetime.strptime(match[3], sber_date_format)

        pp_str = root.xpath('/html/body/p[1]')[0].text_content()
        contract_code = contract_pattern.search(pp_str)[0].replace(' ', '')
        html_tbls = zip(root.xpath('/html/body/p')[1:-1],
                        root.xpath('/html/body/table')[:-1])



        for p, t in html_tbls:
            str = p.text_content()
            str = str.lstrip('\n').split('\n')[0]
            func = table_names.get(str)
            if func:
                df = tbls.get(str)
                if df is not None:
                    tbls[str] = df.append(func(t, contract_code, start_date, end_date))
                else:
                    tbls[str] = func(t, contract_code, start_date, end_date)

        #print(f'Отчет за период {start_date}-{end_date} от {report_date} загружен')

    # Переводим соответствующие столбцы в категориальные
    for mf in [v for v in tbls.values() if v is not None]:
        for h in ['section', 'contract', 'currency', 'sectype']:
            if h in mf.columns:
                mf[h] = mf[h].astype('category')

    secdic = tbls['Справочник Ценных Бумаг**']
    secdic = secdic[~secdic.index.duplicated(keep='last')]

    tbls['Справочник Ценных Бумаг**'] = secdic


    return tbls

def main():
    path = '../Investment-Analysis/sber_reports/'
    arr = [path + f for f in os.listdir(path) if f.endswith('.html')]
    dfs = load_reports(arr)

    pd.set_option('display.max_rows', None)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    aa = dfs['Портфель Ценных Бумаг']
    bb = dfs['Справочник Ценных Бумаг**']

    gg = aa.groupby(by=['date', 'ISIN'])['market_price2'].cumsum()
    #print(aa[~aa.index.duplicated(keep='last')].count())

    #print(bb)

    import matplotlib.pyplot as plt
    import seaborn as sns
    gg = gg.reset_index()
    gg.set_index('ISIN', inplace=True)

    print(bb['sectype'].drop_duplicates().values)
    k = {'Инвестиционный пай иностранного эмитента':'FX',
         'Акция': 'Акции',
         'Облигация':'Облигации',
         'Акция иностранного эмитента':'Акции',
         'Облигация федерального займа':'Облигации',
         'ГДР':'Акции',
         'АДР':'Акции',
         'Акция обыкновенная':'Акции',
         'Акция привилегированная':'Акции'}

    gg['t'] = bb['sectype'].map(k)
    gg = gg.groupby(by=['date','t']).sum()

    #print(gg)
    sns.lineplot(data=gg, x='date', y='market_price2', hue='t')
    plt.show()


if __name__ == "__main__":
    main()