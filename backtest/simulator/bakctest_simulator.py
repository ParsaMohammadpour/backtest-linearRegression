import pandas as pd
import backtrader as bt
import numpy as np
import seaborn as sns
import os
from matplotlib import pyplot as plt

from backtest.strategies.cross_strategy import CrossStrategy, set_cross_strategy_params
from backtest.panda.panda_feed import PandasData


def save_plot(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
    plt.savefig(path)


class BackTestSimulator:
    def __init__(self, samples: dict[str, pd.DataFrame], stock_names: list[str], periods: list[tuple[int, int]],
                 starting_capital: int = 1000000, market_commission: float = 0.0125, save_data: bool = False,
                 base_path: str = 'results/backtest/'):
        self.__samples = {}  # dict[str, pd.DataFrame]
        for stock in stock_names:
            df = samples[stock].copy()
            self.__samples[stock] = df.set_index(pd.DatetimeIndex(df['Date']), inplace=False)
        self.__set_periods(periods)
        self.__stock_names = stock_names
        self.__starting_capital = starting_capital
        self.__market_commission = market_commission
        self.__base_path = base_path
        self.__save_data = save_data
        self.__results = dict(zip(self.__stock_names,
                                  [pd.DataFrame(columns=['cerebro', 'profit-loss', 'strategy-name', 'period'])] * len(
                                      self.__stock_names)))

    def get_samples(self):
        return self.__samples

    def get_results(self):
        return self.__results

    def __period_validation(self, periods: list[tuple[int, int]]):
        for period in periods:
            if period[0] >= period[1]:
                raise ValueError("Short period can't be greater than or equal to long period")

    def __set_periods(self, periods: list[tuple[int, int]]):
        self.__period_validation(periods)
        self.__periods = periods.copy()

    def get_results(self):
        return self.__results.copy()

    def simulate(self, is_log_disabled: bool = False):
        for stock_name in self.__stock_names:
            self.__simulate_stock(stock_name, is_log_disabled)

    def __simulate_stock(self, stock_name: str, is_log_disabled: bool = False):
        for period in self.__periods:
            set_cross_strategy_params(CrossStrategy.sma_strategy_name, period, is_log_disabled)
            self.__simulate_stock_strategy(stock_name, is_log_disabled)
            set_cross_strategy_params(CrossStrategy.ema_strategy_name, period, is_log_disabled)
            self.__simulate_stock_strategy(stock_name, is_log_disabled)

    def __simulate_stock_strategy(self, stock_name: str, is_log_disabled: bool = False):
        if not is_log_disabled:
            print(stock_name)
            print(f'Starting {CrossStrategy.name} for periods: {CrossStrategy.periods}')

        # Create an instance of Cerebro engine
        cerebro = bt.Cerebro()

        # Add the strategy to Cerebro
        cerebro.addstrategy(CrossStrategy)

        # Convert the DataFrame into a Backtrader-compatible data feed
        data_feed = PandasData(dataname=self.__samples[stock_name])

        # Add the data feed to Cerebro
        cerebro.adddata(data_feed)

        # Set initial capital and commissions
        cerebro.broker.setcash(self.__starting_capital)  # starting capital
        cerebro.broker.setcommission(commission=self.__market_commission)  # commission on turnover

        # Print the starting portfolio value
        if not is_log_disabled:
            print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

        # Run the backtest
        results = cerebro.run()

        # Print the final portfolio value
        final_value = cerebro.broker.getvalue()
        if not is_log_disabled:
            print('Final Portfolio Value: %.2f' % final_value)

        # Calculate Profit/Loss and other metrics
        profit_loss = final_value - self.__starting_capital

        if not is_log_disabled:
            print('Net Profit/Loss: %.2f' % profit_loss)

        new_dic = {'cerebro': cerebro,
                   'profit-loss': profit_loss,
                   'strategy-name': CrossStrategy.name,
                   'period': str(CrossStrategy.periods),
                   'trade-no': len(cerebro.broker.orders), }
        new_df = pd.DataFrame(new_dic, index=[0])
        self.__results[stock_name] = (
            self.__results[stock_name].copy() if new_df.empty else new_df.copy() if self.__results[stock_name].empty
            else pd.concat([self.__results[stock_name], new_df], ignore_index=True,
                           axis=0))  # if both DataFrames non empty)

    def best_stock_ma_comparison(self, stock_name: str = None, sma: bool = True, plot: bool = True,
                                 figsize: tuple[int, int] = (15, 7)):
        if stock_name is None:
            stock_name = self.__stock_names
        else:
            if not stock_name in self.__stock_names:
                raise ValueError('Stock name does not exist')
            stock_name = [stock_name]

        res = {}

        for stock in stock_name:
            stock_results = self.__results[stock]
            strategy_name = CrossStrategy.sma_strategy_name if sma else CrossStrategy.ema_strategy_name
            ma_results = stock_results[stock_results['strategy-name'] == strategy_name]

            best_ma = \
                ma_results[ma_results['profit-loss'] == ma_results['profit-loss'].max()].to_dict(orient='records')[0]

            if plot:
                plt.rcParams['figure.figsize'] = figsize
                plt.title(f'{stock} moving average {best_ma['period']}')
                best_ma['cerebro'].plot(iplot=True)[0][0]
                if self.__save_data:
                    save_plot(f'{self.__base_path}cerebro/{stock}/{best_ma["period"]}.png')
            res[stock] = {'period': best_ma['period'],
                          'profit-loss': best_ma['profit-loss'],
                          'profit-loss-percentage': 100 * best_ma['profit-loss'] / self.__starting_capital,
                          'trade-no': best_ma['trade-no'], }

        return res

    def stock_ma_compare_plot(self, stock_name: str = None, sma: bool = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__stock_ma_compare_plot_single(stock_name=stock, sma=sma, figsize=figsize)
        else:
            if stock_name not in self.__stock_names:
                raise ValueError('Stock name does not exist')
            self.__stock_ma_compare_plot_single(stock_name=stock_name, sma=sma, figsize=figsize)

    def __stock_ma_compare_plot_single(self, stock_name: str = None, sma: bool = None,
                                       figsize: tuple[int, int] = (15, 5)):
        stock_results = self.__results[stock_name].copy()
        plot_title = f'{stock_name} moving average comparison'
        strategy_name = 'both'
        colors = []
        if sma is not None:
            strategy_name = CrossStrategy.sma_strategy_name if sma else CrossStrategy.ema_strategy_name
            plot_title = f'{plot_title} for {strategy_name}'
            stock_results = stock_results[stock_results['strategy-name'] == strategy_name]
            colors = np.where(stock_results['profit-loss'] < 0, 'crimson', 'deepskyblue')
        else:
            plot_title = plot_title + '(red and blue are for sam)'
            stock_results['period'] = stock_results['period'].astype(str) + stock_results['strategy-name']
            for ind, row in stock_results.iterrows():
                if row['strategy-name'] == CrossStrategy.sma_strategy_name:
                    if row['profit-loss'] > 0:
                        colors.append('blue')
                    else:
                        colors.append('red')
                else:
                    if row['profit-loss'] > 0:
                        colors.append('green')
                    else:
                        colors.append('orange')
            colors = np.array(colors)

        stock_results.plot.bar('period', 'profit-loss', color=colors, figsize=figsize,
                               title=plot_title)
        if self.__save_data:
            save_plot(f'{self.__base_path}ma-compare/{stock_name}/{strategy_name}.png')
        plt.show()

    def stock_compare_sma_and_ema(self, stock_name: str = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__stock_compare_sma_and_ema_single(stock_name=stock, figsize=figsize)
        else:
            if stock_name not in self.__stock_names:
                raise ValueError('Stock name does not exist')
            self.__stock_compare_sma_and_ema_single(stock_name=stock_name, figsize=figsize)

    def __stock_compare_sma_and_ema_single(self, stock_name: str = '', figsize: tuple[int, int] = (15, 5)):
        stock_results = self.__results[stock_name].copy()
        stock_results['period'] = stock_results['period'].astype(str)
        plt.figure(figsize=figsize)
        sns.barplot(stock_results, x="period", y="profit-loss", hue="strategy-name")
        plt.title(f'{stock_name} moving average comparison')
        if self.__save_data:
            save_plot(f'{self.__base_path}sma-vs-ema/profit-loss/{stock_name}.png')
        plt.show()

    def compare_order_count(self, stock_name: str = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__compare_order_count_single(stock_name=stock, figsize=figsize)
        else:
            if stock_name not in self.__stock_names:
                raise ValueError('Stock name does not exist')
            self.__compare_order_count_single(stock_name=stock_name, figsize=figsize)

    def __compare_order_count_single(self, stock_name: str = None, figsize: tuple[int, int] = (15, 5)):
        df = self.__results[stock_name].copy()
        df['period'] = df['period'].astype(str)
        plt.figure(figsize=figsize)
        sns.barplot(df, x="period", y="trade-no", hue="strategy-name")
        plt.title(f'{stock_name} moving average comparison')
        if self.__save_data:
            save_plot(f'{self.__base_path}sma-vs-ema/order-count/{stock_name}.png')
        plt.show()
