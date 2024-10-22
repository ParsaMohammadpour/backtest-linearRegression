import pandas as pd
import backtrader as bt
from backtest.strategies.cross_strategy import SmaCrossStrategy, EmaCrossStrategy, CrossStrategy
from backtest.panda.panda_feed import PandasData


class BackTestSimulator:
    def __init__(self, samples: dict[str, pd.DataFrame], stock_names: list[str], periods: list[tuple[int, int]],
                 starting_capital: int = 10000,
                 market_commission: float = 0.0125):
        self.__samples = samples
        self.__set_periods(periods)
        self.__stock_names = stock_names
        self.__starting_capital = starting_capital
        self.__market_commission = market_commission
        self.__results = zip(self.__stock_names,
                             pd.DataFrame(columns=['strategy-name', 'profit-loss', 'cerebro', 'strategy']))

    def __period_validation(self, periods: list[tuple[int, int]]):
        for period in periods:
            if period[0] >= period[1]:
                raise ValueError("Short period can't be greater than or equal to long period")

    def __set_periods(self, periods: list[tuple[int, int]]):
        self.__period_validation(periods)
        self.__periods = periods

    def simulate(self, is_log_disabled: bool = False):
        for stock_name in self.__stock_names:
            self.__simulate_stock(stock_name, is_log_disabled)

    def __simulate_stock(self, stock_name: str, is_log_disabled: bool = False):
        for period in self.__periods:
            CrossStrategy.periods = period
            self.__simulate_stock_strategy(stock_name,  SmaCrossStrategy, period, is_log_disabled)
            self.__simulate_stock_strategy(stock_name, EmaCrossStrategy, period, is_log_disabled)

    def __simulate_stock_strategy(self, stock_name: str, strategy:CrossStrategy, period:tuple[int, int], is_log_disabled: bool = False):
        if not is_log_disabled:
            print(f'Starting {strategy.name} for periods: {strategy.periods}')

        # Create an instance of Cerebro engine
        cerebro = bt.Cerebro()

        # Add the strategy to Cerebro
        cerebro.addstrategy(strategy)

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

        self.__results[stock_name].append({'cerebro': cerebro, 'profit-loss': profit_loss, 'strategy': strategy,
                                           'strategy-name': strategy.get_strategy_name(), 'period': period}, ignore_index=True)

    def best_stock_ma_comparison(self, stock_name: str = '', sma: bool = True, plot: bool = True,
                                 figsize: tuple[int, int] = (15, 5)):
        if not stock_name in self.__stock_names:
            raise ValueError('Stock name does not exist')

        stock_results = self.__results[stock_name]
        if sma:
            ma_results = stock_results[stock_results['strategy-name'] == SmaCrossStrategy.name]
        else:
            ma_results = stock_results[stock_results['strategy-name'] == EmaCrossStrategy.name]

        best_ma = \
        ma_results[ma_results['profit-loss'] == ma_results['profit-loss'].max()].to_dict(orient='records')[0]

        if plot:
            best_ma['cerebro'].plot(iplot=False, figsize=figsize)[0][0]

        return best_ma['strategy'].get_period(), best_ma['profit-loss']
