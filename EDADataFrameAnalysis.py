import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from scipy.stats import chi2_contingency, chi2
import seaborn as sns
from scipy import stats
import numpy as np


def show_columns(x):
    return x.columns


class EDADataframeAnalysis:

    def __init__(self):
        print("General_EDA object created")

    def show_dtypes(self, x):
        return x.dtypes

    def Show_Missing(self, x):
        return x.isna().sum()

    def Show_Missing1(self, x):
        return x.isna().sum()

    def Show_Missing2(self, x):
        return x.isna().sum()

    def show_hist(self, x):
        return x.hist()

    def Tabulation(self, x):
        table = pd.DataFrame(x.dtypes, columns=['dtypes'])
        table1 = pd.DataFrame(x.columns, columns=['Names'])
        table = table.reset_index()
        table = table.rename(columns={'index': 'Name'})
        table['No of Missing'] = x.isnull().sum().values
        table['No of Uniques'] = x.nunique().values
        table['Percent of Missing'] = ((x.isnull().sum().values) / (x.shape[0])) * 100
        table['First Observation'] = x.loc[0].values
        table['Second Observation'] = x.loc[1].values
        table['Third Observation'] = x.loc[2].values
        for name in table['Name'].value_counts().index:
            table.loc[table['Name'] == name, 'Entropy'] = round(
                stats.entropy(x[name].value_counts(normalize=True), base=2), 2)
        return table

    def Numerical_variables(self, x):
        Num_var = [var for var in x.columns if x[var].dtypes != "object"]
        Num_var = x[Num_var]
        return Num_var

    def categorical_variables(self, x):
        cat_var = [var for var in x.columns if x[var].dtypes == "object"]
        cat_var = x[cat_var]
        return cat_var

    def impute(self, x):
        df = x.dropna()
        return df

    def imputee(self, x):
        df = x.dropna()
        return df

    def Show_pearsonr(self, x, y):
        result = pearsonr(x, y)
        return result

    def Show_spearmanr(self, x, y):
        result = spearmanr(x, y)
        return result

    def plotly(self, a, x, y):
        fig = px.scatter(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def show_displot(self, x):
        plt.figure(1)
        plt.subplot(121)
        sns.distplot(x)

        plt.subplot(122)
        x.plot.box(figsize=(16, 5))

        plt.show()

    def Show_DisPlot(self, x):
        plt.style.use('fivethirtyeight')
        plt.figure(figsize=(12, 7))
        return sns.distplot(x, bins=25)

    def Show_CountPlot(self, x):
        fig_dims = (18, 8)
        fig, ax = plt.subplots(figsize=fig_dims)
        return sns.countplot(x, ax=ax)

    def plotly_histogram(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def plotly_violin(self, a, x, y):
        fig = px.histogram(a, x=x, y=y)
        fig.update_traces(marker=dict(size=10,
                                      line=dict(width=2,
                                                color='DarkSlateGrey')),
                          selector=dict(mode='markers'))
        fig.show()

    def Show_PairPlot(self, x):
        return sns.pairplot(x)

    def Show_HeatMap(self, x):
        f, ax = plt.subplots(figsize=(15, 15))
        return sns.heatmap(x.corr(), annot=True, ax=ax);

    def label(self, x):
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def label1(self, x):
        le = LabelEncoder()
        x = le.fit_transform(x)
        return x

    def concat(self, x, y, z, axis):
        return pd.concat([x, y, z], axis)

    def dummy(self, x):
        return pd.get_dummies(x)

    def qqplot(self, x):
        return sm.qqplot(x, line='45')

    def PCA(self, x):
        pca = PCA(n_components=8)
        principlecomponents = pca.fit_transform(x)
        principledf = pd.DataFrame(data=principlecomponents)
        return principledf

    def outlier(self, x):
        high = 0
        q1 = x.quantile(.25)
        q3 = x.quantile(.75)
        iqr = q3 - q1
        low = q1 - 1.5 * iqr
        high += q3 + 1.5 * iqr
        outlier = (x.loc[(x < low) | (x > high)])
        return (outlier)

    def check_cat_relation(self, x, y, confidence_interval):
        cross_table = pd.crosstab(x, y, margins=True)
        stat, p, dof, expected = chi2_contingency(cross_table)
        print("Chi_Square Value = {0}".format(stat))
        print("P-Value = {0}".format(p))
        alpha = 1 - confidence_interval
        if p > alpha:
            print(">> Accepting Null Hypothesis <<")
            print("There Is No Relationship Between Two Variables")
        else:
            print(">> Rejecting Null Hypothesis <<")
            print("There Is A Significance Relationship Between Two Variables")
        return p, alpha
