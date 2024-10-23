import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from pylab import rcParams


class LinearRegressionModel:
    r2 = 'r2'
    rmse = 'rmse'
    mape = 'mape'

    def __init__(self, df: pd.DataFrame, stock_name: str, period_max=30, pred_min: float = 0.0, test_size: float = 0.2,
                 cv_size: float = 0.2, target_col: str = 'Close'):
        self.__df = df
        self.__stock_name = stock_name
        self.__period_max = period_max
        self.__pred_min = pred_min
        self.__test_size = test_size  # proportion of dataset to be used as test set
        self.__cv_size = cv_size  # proportion of dataset to be used as cross-validation set
        self.__mape = []
        self.__rmse = []
        self.__r2 = []
        self.__target_col = target_col
        self.__set_df_parts_len()
        self.__set_df_parts()

    def get_mape(self, y_true, y_pred):
        """
        Compute mean absolute percentage error (MAPE)
        """
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    def __set_df_parts_len(self):
        self.__num_cv = int(self.__cv_size * len(self.__df))
        self.__num_test = int(self.__test_size * len(self.__df))
        self.__num_train = len(self.__df) - self.__num_cv - self.__num_test

    def get_df_parts_len(self):
        """
        return: number len of cross validation(cv), test size and train size
        """
        return self.__num_cv, self.__num_test, self.__num_train

    def __set_df_parts(self):
        self.__train = self.__df[:self.__num_train].copy()
        self.__cv = self.__df[self.__num_train:self.__num_train + self.__num_cv].copy()
        self.__train_cv = self.__df[:self.__num_train + self.__num_cv].copy()
        self.__test = self.__df[self.__num_train + self.__num_cv:].copy()

    def get_df_parts(self):
        """
        It gets different len of train, test and cross validation and separates dataframe in relative parts
        :return: train, cross validation(cv), train_cv and test data frames
        """
        return self.__train, self.__cv, self.__train_cv, self.__test

    def plot_regions(self, figsize: tuple[int, int] = (15, 5)):
        rcParams['figure.figsize'] = figsize
        ax = self.__train.plot(x='Date', y=self.__target_col, style='b-', grid=True)
        ax = self.__cv.plot(x='Date', y=self.__target_col, style='y-', grid=True, ax=ax)
        ax = self.__test.plot(x='Date', y=self.__target_col, style='g-', grid=True, ax=ax)
        ax.legend(['train', 'dev', 'test'])
        ax.set_xlabel('Date')
        ax.set_ylabel('USD')
        plt.title(f'{self.__stock_name}')
        plt.show()

    def apply_regression(self):
        regr = LinearRegression(fit_intercept=True)
        pred_list = np.array([])

        for period in range(1, self.__period_max + 1):
            est_list = self.__apply_regression_for_period(period=period, offset=self.__num_train)
            self.__cv.loc[:, 'est' + '-period:' + str(period)] = est_list
            self.__rmse.append(math.sqrt(mean_squared_error(est_list, self.__cv[self.__target_col])))
            self.__r2.append(r2_score(self.__cv[self.__target_col], est_list))
            self.__mape.append(self.get_mape(self.__cv[self.__target_col], est_list))

    def __apply_regression_for_period(self, period: int, offset: int):
        """
        Given a dataframe, get prediction at timestep t using values from t-1, t-2, ..., t-N.
        Inputs
            df         : dataframe with the values you want to predict. Can be of any length.
            target_col : name of the column you want to predict e.g. 'adj_close'
            N          : get prediction at timestep t using values from t-1, t-2, ..., t-N
            pred_min   : all predictions should be >= pred_min
            offset     : for df we only do predictions for df[offset:]. e.g. offset can be size of training set
        Outputs
            pred_list  : the predictions for target_col. np.array of length len(df)-offset.
        """
        # Create linear regression object
        regr = LinearRegression(fit_intercept=True)

        pred_list = []
        df = self.__train_cv

        for i in range(offset, len(self.__train_cv[self.__target_col])):
            X_train = np.array(range(len(self.__train_cv[self.__target_col][i - period:i])))  # e.g. [0 1 2 3 4]
            y_train = np.array(self.__train_cv[self.__target_col][i - period:i])  # e.g. [2944 3088 3226 3335 3436]
            X_train = X_train.reshape(-1, 1)  # e.g X_train =
            # [[0]
            #  [1]
            #  [2]
            #  [3]
            #  [4]]
            # X_train = np.c_[np.ones(N), X_train]              # add a column
            y_train = y_train.reshape(-1, 1)
            #     print X_train.shape
            #     print y_train.shape
            #     print 'X_train = \n' + str(X_train)
            #     print 'y_train = \n' + str(y_train)
            regr.fit(X_train, y_train)  # Train the model
            pred = regr.predict(np.array(period).reshape(1, -1))

            pred_list.append(pred[0][0])  # Predict the footfall using the model

        # If the values are < pred_min, set it to be pred_min
        pred_list = np.array(pred_list)
        pred_list[pred_list < self.__pred_min] = self.__pred_min

        return pred_list

    def plot_param_vs_period(self, param: str = None, figsize: tuple[int, int] = (15, 5)):
        if param == self.rmse:
            y = self.__rmse
        elif param == self.mape:
            y = self.__mape
        elif param == self.r2:
            y = self.__r2
        else:
            raise ValueError(f'Parameter {param} is not valid')

        # Plot RMSE versus N
        matplotlib.rcParams.update({'font.size': 14})
        plt.figure(figsize=figsize, dpi=80)
        plt.plot(range(1, self.__period_max + 1), y, 'x-')
        plt.grid()
        plt.xlabel('N')
        plt.ylabel(param)
        plt.xlim([2, 30])
        plt.title(f'{self.__stock_name} {param} vs period(N)')
        plt.show()