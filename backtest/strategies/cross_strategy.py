import backtrader as bt


# Define the EMA Crossover Strategy
class CrossStrategy(bt.Strategy):
    sma_strategy_name = 'sma'
    ema_strategy_name = 'ema'
    periods = None  # (10, 30) a tuple containing short period and long period
    indicator = None  # bt.indicators.SMA OR bt.indicators.EMA
    name = None
    is_log_disabled = False

    def __init__(self):
        # Initialize the 10-day and 20-day EMA indicators
        self.ema_short = self.indicator(self.data.close, period=self.periods[0])
        self.ema_long = self.indicator(self.data.close, period=self.periods[1])
        self.trade_list = []  # To keep track of trades

    def next(self):
        if not self.position:  # Check if we are not in a position
            if self.ema_short < self.ema_long:  # Buy if short EMA crosses above long EMA
                self.buy()
        else:
            if self.ema_short > self.ema_long:  # Sell if short EMA crosses below long EMA
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
        if order.status in [order.Completed]:
            if order.isbuy():
                if not self.is_log_disabled:
                    print(f"Buy Executed: Price: {order.executed.price}, Size: {order.executed.size}")
            elif order.issell():
                if not self.is_log_disabled:
                    print(f"Sell Executed: Price: {order.executed.price}, Size: {order.executed.size}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            if not self.is_log_disabled:
                print(f"Order Failed: {order.getstatusname()}")

    def stop(self):
        # Print list of trades
        if not self.is_log_disabled:
            print("List of Trades:")
        for trade in self.trade_list:
            if not self.is_log_disabled:
                print(trade)


def set_cross_strategy_params(strategy_name: str = CrossStrategy.sma_strategy_name, periods: tuple[int, int] = (10, 30),
                              is_log_disabled=False):
    CrossStrategy.periods = periods
    if strategy_name == CrossStrategy.sma_strategy_name:
        CrossStrategy.indicator = bt.indicators.SMA
        CrossStrategy.name = CrossStrategy.sma_strategy_name
    elif strategy_name == CrossStrategy.ema_strategy_name:
        CrossStrategy.indicator = bt.indicators.EMA
        CrossStrategy.name = CrossStrategy.ema_strategy_name
    CrossStrategy.is_log_disabled = is_log_disabled
