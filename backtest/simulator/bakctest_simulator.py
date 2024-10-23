import pandas as pd
import backtrader as bt
from matplotlib import pyplot as plt

from backtest.strategies.cross_strategy import CrossStrategy, set_cross_strategy_params
from backtest.panda.panda_feed import PandasData


class BackTestSimulator:
    def __init__(self, samples: dict[str, pd.DataFrame], stock_names: list[str], periods: list[tuple[int, int]],
                 starting_capital: int = 1000000,
                 market_commission: float = 0.0125):
        self.__samples = {}  # dict[str, pd.DataFrame]
        for stock in stock_names:
            df = samples[stock].copy()
            self.__samples[stock] = df.set_index(pd.DatetimeIndex(df['Date']), inplace=False)
        self.__set_periods(periods)
        self.__stock_names = stock_names
        self.__starting_capital = starting_capital
        self.__market_commission = market_commission
        self.__results = dict(zip(self.__stock_names,
                                  [pd.DataFrame(columns=['cerebro', 'profit-loss', 'strategy-name', 'period'])] * len(
                                      self.__stock_names)))

    def __period_validation(self, periods: list[tuple[int, int]]):
        for period in periods:
            if period[0] >= period[1]:
                raise ValueError("Short period can't be greater than or equal to long period")

    def __set_periods(self, periods: list[tuple[int, int]]):
        self.__period_validation(periods)
        self.__periods = periods.copy()

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

        self.__results[stock_name].loc[len(self.__results[stock_name].index)] = {'cerebro': cerebro,
                                                                                 'profit-loss': profit_loss,
                                                                                 'strategy-name': CrossStrategy.name,
                                                                                 'period': CrossStrategy.periods}

    def best_stock_ma_comparison(self, stock_name: str = '', sma: bool = True, plot: bool = True,
                                 figsize: tuple[int, int] = (15, 5)):
        if not stock_name in self.__stock_names:
            raise ValueError('Stock name does not exist')

        stock_results = self.__results[stock_name]
        strategy_name = CrossStrategy.sma_strategy_name if sma else CrossStrategy.ema_strategy_name
        ma_results = stock_results[stock_results['strategy-name'] == strategy_name]

        best_ma = ma_results[ma_results['profit-loss'] == ma_results['profit-loss'].max()].to_dict(orient='records')[0]

        if plot:
            plt.rcParams['figure.figsize'] = figsize
            plt.title(f'{stock_name} moving average {best_ma['period']}')
            best_ma['cerebro'].plot(iplot=True)[0][0]
            plt.show()

        return best_ma['period'], best_ma['profit-loss'], 100 * best_ma['profit-loss'] / self.__starting_capital
