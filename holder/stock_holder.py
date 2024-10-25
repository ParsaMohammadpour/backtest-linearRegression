from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import seaborn as sns
from IPython.core.pylabtools import figsize
from matplotlib.pyplot import figure
import os
import mplfinance as mpf

from datetime import datetime

from dateutil.relativedelta import relativedelta

def save_plot(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(path)


class StockHolder:
    COLUMNS_NAME: dict[str, str] = {"<OPEN>": "Open",
                                    "<HIGH>": "High",
                                    "<LOW>": "Low",
                                    "<CLOSE>": "Close",
                                    "<DTYYYYMMDD>": "Date",
                                    "<VOL>": "Volume",
                                    "<TICKER>": "Name"}
    IGNOR_COLUMNS: list[str] = ['<FIRST>', '<OPENINT>', '<VALUE>', '<LAST>', '<OPENINT>', '<PER>']

    FINAL_COLUMNS: list[str] = ['Close', 'Open', 'High', 'Low']

    def __init__(self, PATH: str = 'stock-samples', DATA_MAX_AGE: int = 5, save_data: bool = False,
                 base_path: str = 'results/data-view/'):
        self.DATA_MAX_AGE = DATA_MAX_AGE
        self.base_path = base_path
        self.__save_data = save_data
        files = [f for f in os.listdir(PATH) if os.path.isfile(PATH + '\\' + f)]
        self.__samples: dict[str, pd.DataFrame] = {}
        self.__stock_names = []
        for file in files:
            self.__stock_names.append(file[:-4])
            self.__samples[file[:-4]] = pd.read_csv(PATH + '\\' + file, index_col=None)

        self.data_cleaning()
        self.__total_df = self.generate_total_df()

    def data_cleaning(self):
        for stock in self.__stock_names:
            df = self.__samples[stock]
            df.drop(self.IGNOR_COLUMNS, axis=1, inplace=True)
            df.rename(columns=self.COLUMNS_NAME, inplace=True)
            df['Date'] = pd.to_datetime(df["Date"], format='%Y%m%d', errors='coerce')
            self.__samples[stock] = df

        self.remove_old_datas()

    def remove_old_datas(self):
        for stock in self.__stock_names:
            df = self.__samples[stock]
            curr_date = datetime.now().date()
            oldest_date = curr_date - relativedelta(years=self.DATA_MAX_AGE)
            oldest_date_dt64 = np.datetime64(oldest_date)
            df = df[df['Date'] > oldest_date_dt64].copy()
            df.sort_values(by='Date', ascending=True, inplace=True, ignore_index=True)
            self.__samples[stock] = df

    def generate_total_df(self):
        df_list = [self.__samples[stock] for stock in self.__stock_names]
        df_total = pd.concat(df_list, axis=0, ignore_index=True)
        return df_total

    def describe_plot(self, stock_name: str = None, figsize: tuple = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__describe_plot_single(stock, figsize)
        else:
            self.__describe_plot_single(stock_name, figsize)

    def __describe_plot_single(self, stock_name: str, figsize: tuple = (15, 5)):
        df = self.__samples[stock_name]
        print(f'stock stats:')
        print(df.describe)

        df = df.drop('Volume', axis=1, inplace=False).copy()
        color = {
            "boxes": "DarkGreen",
            "whiskers": "DarkOrange",
            "medians": "DarkBlue",
            "caps": "Gray",
        }
        df.plot.box(color=color, sym="r+", figsize=figsize)
        plt.title(f'{stock_name} stats')
        if self.__save_data:
            save_plot(f'{self.base_path}stats/{stock_name}.png')
        plt.show()

    def describe_together_single_col(self, col: str = None):
        if col is None:
            col = self.FINAL_COLUMNS
        else:
            if col not in self.FINAL_COLUMNS:
                raise ValueError(f'col {col} must be in {self.FINAL_COLUMNS}')
            col = [col]
        for column in col:
            group_by_res = []
            for stock in self.__stock_names:
                df = self.__samples[stock]
                year_df = df.groupby(df.Date.dt.year)[self.FINAL_COLUMNS].mean()
                year_df['stock-name'] = pd.Series([stock] * len(year_df), index=year_df.index)
                group_by_res.append(year_df)
            total_year_group_by_df = pd.concat(group_by_res, axis=0, ignore_index=True)
            sns.catplot(data=total_year_group_by_df, x='stock-name', y=column, kind="box", height=5, aspect=2)
            plt.title(f'comparing stock statistical of {column} price')
            if self.__save_data:
                save_plot(f'{self.base_path}stock-stats-comparison/{column}.png')
            plt.show()

    def describe_together_all_cols(self):
        group_by_res = []
        for stock in self.__stock_names:
            df = self.__samples[stock]
            year_df = df.groupby(df.Date.dt.year)[self.FINAL_COLUMNS].mean()
            year_df['stock-name'] = pd.Series([stock] * len(year_df), index=year_df.index)
            group_by_res.append(year_df)
        total_year_group_by_df = pd.concat(group_by_res, axis=0, ignore_index=True)
        for col in self.FINAL_COLUMNS:
            sns.catplot(data=total_year_group_by_df, x='stock-name', y=col, kind="box", height=5, aspect=2)
            plt.title(f'comparing stock statistical of {col} price')
            if self.__save_data:
                save_plot(f'{self.base_path}stock-stats-comparison/{col}.png')
            plt.show()

    def close_open_diff(self, stock_name: str = None, period_in_day: int = 100, is_head: bool = True,
                        figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__close_open_diff_single(stock, period_in_day, is_head)
        else:
            if not stock_name in self.__samples:
                raise ValueError("Stock " + stock_name + " not found.")
            self.__close_open_diff_single(stock_name, period_in_day, is_head, figsize)

    def __close_open_diff_single(self, stock_name: str, period_in_day: int = 100, is_head: bool = True,
                                 figsize: tuple[int, int] = (15, 5)):
        df = self.__samples[stock_name]
        plt.figure(figsize=figsize)
        target_df = df.tail(period_in_day) if is_head else df.tail(period_in_day)
        plt.bar(
            target_df.index, target_df["Close"] - target_df["Open"],
            color=np.where(target_df["Close"] - target_df["Open"] < 0, 'crimson', 'lightgreen'))
        period = 'first' if is_head else 'last'
        plt.title(f'{stock_name} {period} {period_in_day} closing - opening price')
        plt.xlabel('index')
        plt.ylabel('price difference')
        if self.__save_data:
            save_plot(f'{self.base_path}close-open-diff/{stock_name}/{period}.png')
        plt.show()

    def year_mean(self, stock_name: str = None, col: str = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            stock_name = self.__stock_names
        else:
            if not stock_name in self.__samples:
                raise ValueError(f'stock {stock_name} not in samples')
            stock_name = [stock_name]
        if col is None:
            col = self.FINAL_COLUMNS
        else:
            if not col in self.FINAL_COLUMNS:
                raise ValueError(f'col {col} must be in {self.FINAL_COLUMNS}')
            col = [col]
        for column in col:
            for stock in stock_name:
                self.__year_mean_single(stock, column, figsize)


    def __year_mean_single(self, stock_name: str, col: str, figsize: tuple[int, int] = (15, 5)):
        df = self.__samples[stock_name]
        year_df = df.groupby(df.Date.dt.year)[[col]].mean()
        year_df.plot.bar(y=col, figsize=figsize)
        plt.title(f'{stock_name} year mean')
        if self.__save_data:
            save_plot(f'{self.base_path}year-mean/{stock_name}/{col}.png')
        plt.show()

    def year_mean_compare(self, col: str = None, figsize: tuple[int, int] = (12, 5)):
        if col is None:
            for column in self.FINAL_COLUMNS:
                self.__year_mean_compare_single(col=column, figsize=figsize)
        else:
            if col not in self.FINAL_COLUMNS:
                raise ValueError(f'column {col} not in {self.FINAL_COLUMNS}')
            self.__year_mean_compare_single(col=col, figsize=figsize)

    def __year_mean_compare_single(self, col: str = None, figsize: tuple[int, int] = (12, 5)):
        group_df = self.__total_df.groupby([self.__total_df.Date.dt.year, 'Name'])[[col]].mean()
        plt.figure(figsize=figsize)
        sns.barplot(data=group_df, x='Date', y=col, hue='Name')
        if self.__save_data:
            save_plot(f'{self.base_path}year-mean-comparison/{col}.png')

    def year_mean_diff(self, stock_name: str = None, col: str = None, figsize: tuple[int, int] = (15, 5)):
        if col is not None:
            if col not in self.FINAL_COLUMNS:
                raise ValueError(f'column {col} not in {self.FINAL_COLUMNS}')
            col = [col]
        else:
            col = self.FINAL_COLUMNS
        if stock_name is not None:
            if stock_name not in self.__samples:
                raise ValueError("Stock " + stock_name + " not found.")
            stock_name = [stock_name]
        else:
            stock_name = self.__stock_names

        for stock in stock_name:
            for column in col:
                self.__year_mean_diff_single(stock, column, figsize)


    def __year_mean_diff_single(self, stock_name: str, col: str, figsize: tuple[int, int] = (15, 5)):
        df = self.__samples[stock_name]
        total_mean = df[col].mean() if len(df) > 1 else 0
        year_df = df.groupby(df.Date.dt.year)[[col]].mean()

        colors = np.where(year_df[col] - total_mean < 0, 'crimson', 'deepskyblue')
        plt.figure(figsize=figsize)
        plt.bar(x=year_df.index, height=year_df[col] - total_mean, color=colors, )
        plt.title(f'{stock_name} year - mean')
        plt.xlabel('year')
        plt.ylabel(f'{col.lower()} price')
        if self.__save_data:
            save_plot(f'{self.base_path}year-mean-diff-comparison/{stock_name}/{col}.png')
        plt.show()

    def data_plot_single_stock(self, stock_name: str = None, is_number_index: bool = True, col: str = None,
                               figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            stock_name = self.__stock_names
        else:
            if not stock_name in self.__stock_names:
                raise ValueError(f'stock {stock_name} not in samples')
            stock_name = [stock_name]
        if col is None:
            col = self.FINAL_COLUMNS
        else:
            if not col in self.FINAL_COLUMNS:
                raise ValueError(f'column {col} not in {self.FINAL_COLUMNS}')
            col = [col]

        for column in col:
            for stock in stock_name:
                self.__data_plot_single_stock(stock_name=stock, col=column, is_number_index=is_number_index,
                                              figsize=figsize)

    def __data_plot_single_stock(self, stock_name: str, is_number_index: bool = True, col: str = None,
                                 figsize: tuple[int, int] = (15, 5)):
        df = self.__samples[stock_name]
        plt.figure(figsize=figsize)
        x = df.index if is_number_index else df['Date']
        sns.lineplot(data=df, x=x, y=col, color='firebrick')
        sns.despine()
        plt.title(f"The Stock {col} Price of {stock_name}", size='x-large', color='blue')
        if self.__save_data:
            save_plot(f'{self.base_path}single-data-plot/{stock_name}/{col}.png')
        plt.show()

    def data_plot_compare(self, col: str = None, figsize: tuple[int, int] = (15, 5)):
        if col is None:
            col = self.FINAL_COLUMNS
        else:
            if not col in self.FINAL_COLUMNS:
                raise ValueError(f'column {col} not in {self.FINAL_COLUMNS}')
            col = [col]
        for column in col:
            plt.figure(figsize=figsize)
            sns.lineplot(data=self.__total_df, x=self.__total_df.Date, y=column, hue='Name', color='firebrick')
            sns.despine()
            plt.title(f"The Stock {column} Price", size='x-large', color='blue')
            if self.__save_data:
                save_plot(f'{self.base_path}data-compare/{column}.png')

    def mpf_last_days_view(self, stock_name: str = None, days: int = 100, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__mpf_last_days_view_single(stock, days, figsize)
        else:
            if not stock_name in self.__samples:
                raise ValueError(f'stock {stock_name} not in samples')
            self.__mpf_last_days_view_single(stock_name, days, figsize)

    def __mpf_last_days_view_single(self, stock_name: str = None, days: int = 100, figsize: tuple[int, int] = (15, 5)):
        df = self.__samples[stock_name].copy()
        df.set_index('Date').reset_index(drop=True, inplace=True)
        df.index = pd.DatetimeIndex(df['Date'])
        if self.__save_data:
            path = f'{self.base_path}mpf/{stock_name}/'
            dir = os.path.dirname(path)
            if not os.path.exists(dir):
                os.makedirs(dir)
            mpf.plot(df.tail(days), type='candle', style='charles', title=f'{stock_name} Candlestick Chart',
                     volume=True, figsize=figsize, savefig=path+f'candle-{days}.png')
            mpf.plot(df.tail(days), type='candle', style='charles', title=f'{stock_name} Candlestick Chart',
                     volume=True, figsize=figsize, savefig=path+f'pnf-{days}.png')
            mpf.plot(df.tail(days), type='candle', style='charles', title=f'{stock_name} Candlestick Chart',
                     volume=True, figsize=figsize, savefig=path + f'renko-{days}.png')
        else:
            mpf.plot(df.tail(days), type='candle', style='charles', title=f'{stock_name} Candlestick Chart',
                     volume=True, figsize=figsize)
            plt.show()

    def get_samples(self):
        return self.__samples.copy()

    def get_stock_names(self):
        return self.__stock_names.copy()
