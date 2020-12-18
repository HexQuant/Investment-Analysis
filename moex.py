import pandas as pd
import requests
import datetime
from io import StringIO

moex_url = "https://iss.moex.com/"

moex_max_limit = 100
moex_header = 1
moex_footer = 8

def get_indices():
    df = pd.DataFrame()
    r = requests.get(
        f"{moex_url}iss/statistics/engines/stock/markets/index/analytics.csv?limit={moex_max_limit}")
    data = StringIO(r.text)
    df = pd.read_csv(data, delimiter=";", skiprows=2, index_col="indexid")
    df["shortname"] = df["shortname"].str.strip()
    return df


def get_stocks_in_indices(name):

    url = f"{moex_url}iss/statistics/engines/stock/markets/index/analytics/{name}.csv"

    r = requests.get(f"{url}?iss.only=analytics.cursor")
    data = StringIO(r.text)
    info = pd.read_csv(data, delimiter=";", skiprows=2)

    df = pd.DataFrame()
    start = 0
    while start <= info["TOTAL"][0]:
        r = requests.get(f"{url}?iss.only=analytics&limit={moex_max_limit}&start={start}")
        print(f"{url}?iss.only=analytics&limit={moex_max_limit}&start={start}")
        data = StringIO(r.text)

        if df.empty:
            df = pd.read_csv(data, delimiter=";", skiprows=2, index_col="ticker")
        else:
            df = pd.concat(
                [pd.read_csv(data, delimiter=";", skiprows=2, index_col="ticker"), df])

        start = start + moex_max_limit

    df.drop(columns=["indexid", "secids", "tradedate"], inplace=True)
    return df



def get_stocks():

    url = (f"{moex_url}iss/engines/stock/markets/shares/boards/TQBR/securities.csv?iss.only=securities",
           f"{moex_url}iss/engines/stock/markets/shares/boards/TQBR/securities.csv?iss.only=marketdata")

    df_join = pd.DataFrame()
    for i in range(0, len(url)):
        r = requests.get(url[i])
        data = StringIO(r.text)
        df = pd.read_csv(data, delimiter=";", skiprows=2, index_col="SECID") # decimal="."
        if df_join.empty:
            df_join = df
        else:
            df_join = df_join.join(df, lsuffix='_left', rsuffix='_right', how='left', on="SECID")

    df_join.dropna(axis='columns', how='all', inplace=True)
    return df_join

def get_stock_marketdata(ticker, first = None, end = None, limit = moex_max_limit):

    params = []
    if first != None:
        params.append(f'from={first}')
    if end != None:
        params.append(f'till={end}')
    params_str = '&'.join(params)
    url = f'{moex_url}iss/history/engines/stock/markets/shares/securities/{ticker}.csv?iss.only=marketdata&limit={limit}&{params_str}&'
    print(url)

    df = pd.read_csv(url, skiprows=moex_header, skipfooter=moex_footer, sep=';')
    recived_rows = df.shape[0]
    start = 0
    while recived_rows == limit:
        start += limit
        join_url = url+f'start={start}'
        print(join_url)
        join_df = pd.read_csv(join_url, skiprows=moex_header, skipfooter=moex_footer, sep=';')
        recived_rows = join_df.shape[0]
        df = pd.concat([df, join_df], ignore_index=True)

    return df


def main():

    df = get_stock_marketdata('SBMX')
    print(df)
    df['TRADEDATE'] = pd.to_datetime(df['TRADEDATE'])
    per_bb = df['TRADEDATE'].dt.to_period("M")
    g = df.groupby(per_bb)['WAPRICE'].mean()
    print(g)



if __name__ == "__main__":
    main()
