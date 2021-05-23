import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

class Attribute_Information():

    def __init__(self):
        print("Attribute Information object created")

    def Column_information(self, data):
        data_info = pd.DataFrame(
            columns=['No of observation',
                     'No of Variables',
                     'No of Numerical Variables',
                     'No of Factor Variables',
                     'No of Categorical Variables',
                     'No of Logical Variables',
                     'No of Date Variables',
                     'No of zero variance variables'])

        data_info.loc[0, 'No of observation'] = data.shape[0]
        data_info.loc[0, 'No of Variables'] = data.shape[1]
        data_info.loc[0, 'No of Numerical Variables'] = data._get_numeric_data().shape[1]
        data_info.loc[0, 'No of Factor Variables'] = data.select_dtypes(include='category').shape[1]
        data_info.loc[0, 'No of Logical Variables'] = data.select_dtypes(include='bool').shape[1]
        data_info.loc[0, 'No of Categorical Variables'] = data.select_dtypes(include='object').shape[1]
        data_info.loc[0, 'No of Date Variables'] = data.select_dtypes(include='datetime64').shape[1]
        data_info.loc[0, 'No of zero variance variables'] = data.loc[:, data.apply(pd.Series.nunique) == 1].shape[1]

        data_info = data_info.transpose()
        data_info.columns = ['value']
        data_info['value'] = data_info['value'].astype(int)

        return data_info

    def __get_missing_values(self, data):
        # Getting sum of missing values for each feature
        missing_values = data.isnull().sum()
        # Feature missing values are sorted from few to many
        missing_values.sort_values(ascending=False, inplace=True)
        # Returning missing values
        return missing_values

    def __iqr(self, x):
        return x.quantile(q=0.75) - x.quantile(q=0.25)

    def __outlier_count(self, x):
        upper_out = x.quantile(q=0.75) + 1.5 * self.__iqr(x)
        lower_out = x.quantile(q=0.25) - 1.5 * self.__iqr(x)
        return len(x[x > upper_out]) + len(x[x < lower_out])

    def num_count_summary(self, df):
        df_num = df._get_numeric_data()
        data_info_num = pd.DataFrame()
        i = 0
        for c in df_num.columns:
            data_info_num.loc[c, 'Negative values count'] = df_num[df_num[c] < 0].shape[0]
            data_info_num.loc[c, 'Positive values count'] = df_num[df_num[c] > 0].shape[0]
            data_info_num.loc[c, 'Zero count'] = df_num[df_num[c] == 0].shape[0]
            data_info_num.loc[c, 'Unique count'] = len(df_num[c].unique())
            data_info_num.loc[c, 'Negative Infinity count'] = df_num[df_num[c] == -np.inf].shape[0]
            data_info_num.loc[c, 'Positive Infinity count'] = df_num[df_num[c] == np.inf].shape[0]
            data_info_num.loc[c, 'Missing Percentage'] = df_num[df_num[c].isnull()].shape[0] / df_num.shape[0]
            data_info_num.loc[c, 'Count of outliers'] = self.__outlier_count(df_num[c])
            i = i + 1
        return data_info_num

    def categorical_variables(self, x):
        cat_var = [var for var in x.columns if x[var].dtypes == "object"]
        cat_var = x[cat_var]
        return cat_var

    def impute(self, x):
        df = x.dropna()
        return df
    def imputer(self,x):
        imputer = KNNImputer(n_neighbors=2)
        x[:] = imputer.fit_transform(x)
        return x



    def statistical_summary(self, df):
        df_num = df._get_numeric_data()
        data_stat_num = pd.DataFrame()
        try:
            data_stat_num = pd.concat([df_num.describe().transpose(),
                                       pd.DataFrame(df_num.quantile(q=0.10)),
                                       pd.DataFrame(df_num.quantile(q=0.90)),
                                       pd.DataFrame(df_num.quantile(q=0.95))], axis=1)
            data_stat_num.columns = ['count', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', '10%', '90%', '95%']
        except:
            pass
        return data_stat_num
