import datetime
import backtrader as bt

# Create a custom pandas feed to use with Backtrader
class PandasData(bt.feeds.PandasData):
    params = (
        #('fromdate', datetime.datetime(218, 1, 1)),
        #('todate', datetime.datetime(2023, 1, 1)),
        ('open', 'Open'),
        ('high', 'High'),
        ('low', 'Low'),
        ('close', 'Close'),
        ('volume', 'Volume'),
        #('openinterest', None),  # No open interest in Yahoo data
    )