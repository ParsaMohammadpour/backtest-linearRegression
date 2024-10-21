import backtrader as bt


# Define the EMA (Exponential Moving Average) Crossover Strategy
class CrossStrategy(bt.Strategy):
    params = (('short_period', 10), ('long_period', 20))

    def __init__(self, periods: tuple[tuple[str, int], tuple[str, int]] = params, bt_strategy: bt.indicators = bt.indicators.SMA,
                 is_log_disabled: bool = False):
        # validation
        if periods is None:
            raise ValueError('EmaCrossStrategy requires period tuple')
        if len(periods) != 2:
            raise ValueError('EmaCrossStrategy requires two tuple')
        if periods.short_period >= periods.long_period:
            raise ValueError('short_period cannot be greater than or equal to long_period')
        # Initialize the indicators
        self.__bt_strategy = bt_strategy
        self.ema_short = bt_strategy(self.data.close, period=self.params.short_period)
        self.ema_long = bt_strategy(self.data.close, period=self.params.long_period)
        self.trade_list = []  # To keep track of trades
        self.is_log_disabled = is_log_disabled
        self.__periods = periods

    def next(self):
        if not self.position:  # Check if we are not in a position
            if self.ema_short < self.ema_long:  # Buy if short EMA crosses below long EMA
                self.buy()
        else:
            if self.ema_short > self.ema_long:  # Sell if short EMA crosses above long EMA
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

    def is_sma(self):
        return self.bt_strategy is bt.indicators.SMA

    def get_period(self):
        return self.__periods
