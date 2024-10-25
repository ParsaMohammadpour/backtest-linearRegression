# AlgoTrading 1st assignment
&emsp; In this assignment we have to choose two of specified stocks and apply following operations on them. Before that I have plotted some stats and data in order to have a visualization about the data. After that we jump into these steps. These steps are as follows:
<br/>

### 1- Backtest:
&emsp; Applying backtesting method for them and for different pair of simple moving average (SMA) base like (4,8) and ... and then compare them with each other and make a conclusion on the best pair of numbers and plotting them. Also saving plots in **results** folder in order to have them in specified folders. 
<br/>

### 2- Linear Regreesion:
&emsp; Applying LinearRegression on the input and validating it by 20% of the output. Here we set different period for this regression. It means that everytime we get N (period length) data, and train a linear regression model over it using the same Linearregression object and predicting the prediction for the next day (index to be more accurate). Then we calculate some functions like mape (Mean absolute percentage error), rsme (root mean squared error) and r2 (R squared) to compare the result of different periods.
<br/>
<br/>
<br/>
For running this notebook you simpley can install requirements and then run this notebook. Note that if you set the save_data property to True, it will generate and save the results in the base_path.
One cell has been commented, run that cell at the end of the execution of all notebook. Because that uses js and changes some properties in matplotlib that makes problem for other plots that are going to be plotted.
And if you only want to focus on one stock, you can simply set SAMPLE_STOCK_NAME to that stock file name, and see the results for that specific stock. And you can do the same with the SAMPLE_COLUMN_NAME, if you are only interested in the closing price of a stock and you only want to see them, you can set that value (value of the SAMPLE_COLUMN_NAME) to Close.
<br/>
You can Also change other properties like MARKET_COMMISSION, linear regression periods, linear regression train, test percentage and ... to see results relevant to them. 
