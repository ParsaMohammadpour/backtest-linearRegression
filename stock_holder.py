from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from matplotlib.pyplot import figure
import os

from pandas.core.sample import sample

sns.set_style()
import mplfinance as mpf

from datetime import datetime

from dateutil.relativedelta import relativedelta


class StockHolder:
    COLUMNS_NAME: dict[str, str] = {"<OPEN>": "Open", "<HIGH>": "High", "<LOW>": "Low", "<CLOSE>": "Close",
                                    "<DTYYYYMMDD>": "Date",
                                    "<VOL>": "Volume"}
    IGNOR_COLUMNS: list[str] = ['<FIRST>', '<OPENINT>', '<VALUE>', '<LAST>', '<OPENINT>']

    FINAL_COLUMNS: list[str] = ['Close', 'Open', 'High', 'Low']

    def __init__(self, PATH: str = 'stock-samples', DATA_MAX_AGE: int = 5):
        self.DATA_MAX_AGE = DATA_MAX_AGE
        files = [f for f in os.listdir(PATH) if os.path.isfile(PATH + '\\' + f)]
        self.samples : dict[str, pd.DataFrame] = {}
        self.stock_names = []
        for file in files:
            self.stock_names.append(file[:-4])
            self.samples[file[:-4]] = pd.read_csv(PATH + '\\' + file, index_col=None)

        self.data_cleaning()
        self.total_df = self.generate_total_df()

    def data_cleaning(self):
        for stock in self.stock_names:
            df = self.samples[stock]
            df.drop(self.IGNOR_COLUMNS, axis=1, inplace=True)
            df.rename(columns=self.COLUMNS_NAME, inplace=True)
            df['Date'] = pd.to_datetime(df["Date"], format='%Y%m%d', errors='coerce')
            self.samples[stock] = df

        self.remove_old_datas()


    def remove_old_datas(self):
        for stock in self.stock_names:
            df = self.samples[stock]
            curr_date = datetime.now().date()
            oldest_date = curr_date - relativedelta(years=self.DATA_MAX_AGE)
            oldest_date_dt64 = np.datetime64(oldest_date)
            df = df[df['Date'] > oldest_date_dt64].copy()
            df.sort_values(by='Date', ascending=True, inplace=True, ignore_index=True)
            self.samples[stock] = df

    def generate_total_df(self):
        df_list = [self.samples[stock] for stock in self.stock_names]
        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total

    def describe_plot(self):
        for stock in self.stock_names:
            df = self.samples[stock]
            print(f'stock stats:')
            print(df.describe)

            df = df.drop('Volume', axis=1, inplace=False).copy()
            color = {
                "boxes": "DarkGreen",
                "whiskers": "DarkOrange",
                "medians": "DarkBlue",
                "caps": "Gray",
            }
            df.plot.box(color=color, sym="r+", figsize=(13, 5))
            plt.title(f'{stock} stats')
            plt.show()

    def describe_together(self):
        group_by_res = []
        for stock in self.stock_names:
            df = self.samples[stock]
            year_df = df.groupby(df.Date.dt.year)[self.FINAL_COLUMNS].mean()
            year_df['stock-name'] = pd.Series([stock] * len(year_df), index=year_df.index)
            group_by_res.append(year_df)
        total_year_group_by_df = pd.concat(group_by_res, axis=0, ignore_index=True)
        for col in self.FINAL_COLUMNS:
            sns.catplot(data=total_year_group_by_df, x='stock-name', y=col, kind="box", height=5, aspect=2)
            plt.title(f'comparing stock statistical of {col} price')


    def close_open_diff(self, period_in_day: int = 100, is_head: bool = True):
        for stock in self.stock_names:
            df = self.samples[stock]
            target_df = df.tail(period_in_day) if is_head else df.tail(period_in_day)
            plt.bar(
                target_df.index, target_df["Close"] - target_df["Open"],
                color=np.where(target_df["Close"] - target_df["Open"] < 0, 'crimson', 'lightgreen'))
            period = 'first' if is_head else 'last' 'last'
            plt.title(f'{stock} {period} {period_in_day} closing - opening price')
            plt.xlabel('index')
            plt.ylabel('price difference')
            plt.show()

    def year_mean(self, col:str='Close'):
        group_by_res = []
        for stock in self.stock_names:
            df = self.samples[stock]
            year_df = df.groupby(df.Date.dt.year)[[col]].mean()
            group_by_res.append(year_df)
            year_df.plot.bar(y=col)
            plt.title(f'{stock} each year mean {col} price')
            plt.show()

        plt.figure(figsize=(13,5))
        # making new dictionary to generate a new dataframe in order to have group bar plots against stacked bar plots

        for i in range(len(self.stock_names)):
            y = group_by_res[i][col].values
            if len(y) < self.DATA_MAX_AGE + 1:
                y = [*list(range(self.DATA_MAX_AGE + 1 - len(y))) , *y]


        merged_mean_df = pd.DataFrame(merged_mean)
        merged_mean_df.plot.bar(label=self.stock_names)
        plt.xlabel('year')
        plt.ylabel(col.lower() + " price")
        plt.title(f'difference of mean price of stock in last years of {col.lower()} price')
        plt.legend(loc='best')
        plt.show()



    def year_mean_compare(self, col:str='Close'):
        for stock in self.stock_names:
            df = self.samples[stock]
            total_mean = df.price.mean()
            year_df = df.groupby(df.Date.dt.year)[[col]].mean()
            colors = np.where(year_df.price - total_mean < 0, 'crimson', 'deepskyblue')
            plt.bar(x=year_df.index, height=year_df.price - total_mean, color=colors)
            plt.title(f'{stock} each year mean price - total price of {col.lower()} price')
            plt.xlabel('year')
            plt.ylabel(f'{col.lower()} price')
            plt.legend(loc='best')
            plt.show()