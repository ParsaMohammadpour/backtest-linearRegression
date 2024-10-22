import backtrader as bt


# Define the MA (Moving Average) Crossover Strategy Base
class CrossStrategy(bt.Strategy):
    periods = None # short_period, long_period and should be set before being assigned as strategy to
    name= 'ma'

    def __init__(self):
        # validation
        if self.periods is None:
            raise ValueError('maCrossStrategy requires period tuple')
        if len(self.periods) != 2:
            raise ValueError('maCrossStrategy requires two tuple')
        if self.periods[0] >= self.periods[1]:
            raise ValueError('short_period cannot be greater than or equal to long_period')

        # Initialize the indicators in children
        self.short_period_ma = None
        self.long_period_ma = None

        self.trade_list = []  # To keep track of trades

    def next(self):
        if not self.position:  # Check if we are not in a position
            if self.short_period_ma < self.long_period_ma:  # Buy if short EMA crosses below long EMA
                self.buy()
        else:
            if self.short_period_ma > self.long_period_ma:  # Sell if short EMA crosses above long EMA
                self.sell()

    def notify_trade(self, trade):
        if trade.isclosed:
            exit_price = None
            if trade.size != 0:
                exit_price = trade.price + trade.pnlcomm / trade.size  # Calculate exit price
            else:
                exit_price = trade.price  # Set exit price to entry price if size is zero (or handle as needed)

            # Log the trade details when a trade is closed
            trade_details = {
                'Entry Price': trade.price,  # Entry price of the trade
                'Exit Price': exit_price,  # Calculated exit price
                'Size': trade.size,  # Size of the trade
                'Profit/Loss': trade.pnlcomm  # Net PnL after commission
            }
            self.trade_list.append(trade_details)

    def notify_order(self, order):
        if self.is_log_disabled:
            return
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f"Buy Executed: Price: {order.executed.price}, Size: {order.executed.size}")
            elif order.issell():
                print(f"Sell Executed: Price: {order.executed.price}, Size: {order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            print("Order Failed")

    def stop(self):
        if self.is_log_disabled:
            return
        # Print list of trades
        print("List of Trades:")
        for trade in self.trade_list:
            print(trade)
        print("\n")

    def get_period(self):
        return self.__periods

    def get_strategy_name(self):
        pass


# Define the SMA (Simple Moving Average) Crossover Strategy
class SmaCrossStrategy(CrossStrategy):
    name = 'sma'

    def __init__(self):
        super().__init__()
        self.short_period_ma = bt.indicators.SMA(self.data.close, period=self.params[0])
        self.long_period_ma = bt.indicators.SMA(self.data.close, period=self.params[1])

    def get_strategy_name(self):
        return self.name


# Define the EMA (Exponential Moving Average) Crossover Strategy
class EmaCrossStrategy(CrossStrategy):
    name = 'ema'

    def __init__(self):
        super().__init__()
        self.short_period_ma = bt.indicators.EMA(self.data.close, period=self.params[0])
        self.long_period_ma = bt.indicators.EMA(self.data.close, period=self.params[1])

    def get_strategy_name(self):
        return self.name
