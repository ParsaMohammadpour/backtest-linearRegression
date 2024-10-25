from regression.LinearRegressionModel import LinearRegressionModel
import pandas as pd
import os



def create_directory(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)


class LrSimulator:
    def __init__(self, samples: dict[str, pd.DataFrame], stock_names: list[str], period_max=30, pred_min: float = 0.0,
                 test_size: float = 0.2, cv_size: float = 0.2, target_col: str = 'Close', save_data: bool = False,
                 base_path: str = 'results/linear-regression/'):
        self.__samples = samples.copy()
        self.__stock_names = stock_names.copy()
        self.__period_max = period_max
        self.__pred_min = pred_min
        self.__test_size = test_size
        self.__cv_size = cv_size
        self.__target_col = target_col
        self.__save_data = save_data
        self.__base_path = base_path
        self.__generate_slr_simulators()

    def __generate_slr_simulators(self):
        self.__lr_sims = {}
        for stock in self.__stock_names:
            self.__lr_sims[stock] = LinearRegressionModel(df=self.__samples[stock], stock_name=stock,
                                                          period_max=self.__period_max, pred_min=self.__pred_min,
                                                          test_size=self.__test_size, cv_size=self.__cv_size,
                                                          target_col=self.__target_col, base_path=self.__base_path,
                                                          save_data=self.__save_data)

    def plot_regions(self, stock_name: str = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__lr_sims[stock].plot_regions(figsize=figsize)
        else:
            if not stock_name in self.__stock_names:
                raise ValueError('Socket name must be in stock list')
            self.__lr_sims[stock_name].plot_regions(figsize=figsize)

    def apply_regressions(self):
        for stock in self.__stock_names:
            self.__lr_sims[stock].apply_regression()

    def plot_param_vs_period(self, stock_name: str = None, param: str = None, figsize: tuple[int, int] = (15, 5)):
        if stock_name is None:
            for stock in self.__stock_names:
                self.__lr_sims[stock].plot_param_vs_period(param=param, figsize=figsize)
        else:
            if not stock_name in self.__stock_names:
                raise ValueError('Socket name must be in stock list')
            self.__lr_sims[stock_name].plot_param_vs_period(param=param, figsize=figsize)

    def predict_future_days(self, days: int = 10, period: int = 5, figsize: tuple[int, int] = (15, 5),
                            plot_list: list[str] = None):
        """
        :param days:
        :param period:
        :param figsize:
        :param plot_list:
        :return: today's price and estimate for next 10 days
        """
        res = {}
        plot_list = self.__stock_names if plot_list is None else plot_list
        for stock in self.__stock_names:
            plot = stock in plot_list
            predictions = self.__lr_sims[stock].predict_future_days(days=days, period=period, plot=plot,
                                                                    figsize=figsize)
            res[stock] = predictions.flatten()

        df = pd.DataFrame(res)
        if self.__save_data:
            path = f'{self.__base_path}predictions-csv/estimate.csv'
            create_directory(path)
            df.to_csv(path, index=False)
        return df
