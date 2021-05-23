import streamlit as st
st.set_option('deprecation.showPyplotGlobalUse', False)

import itertools
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy import stats
import matplotlib.pyplot as plt
import plotly.express as px
import matplotlib
from sklearn.metrics import mean_squared_error, accuracy_score

from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from scipy.stats import chi2_contingency, chi2
import statsmodels.api as sm
from scipy.stats import spearmanr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
from scipy.stats import anderson
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB



# image = Image.open('cover.jpg')
matplotlib.use("Agg")

class Data_Base_Modelling():
    def __init__(self):
        print("General_EDA object created")

    def Label_Encoding(self, x):
        category_col = [var for var in x.columns if x[var].dtypes == "object"]
        labelEncoder = preprocessing.LabelEncoder()
        mapping_dict = {}
        for col in category_col:
            x[col] = labelEncoder.fit_transform(x[col])
            le_name_mapping = dict(zip(labelEncoder.classes_, labelEncoder.transform(labelEncoder.classes_)))
            mapping_dict[col] = le_name_mapping
        return mapping_dict

    def IMpupter(self, x):
        imp_mean = IterativeImputer(random_state=0)
        x = imp_mean.fit_transform(x)
        x = pd.DataFrame(x)
        return x

    def dummy(self, x):
        return pd.get_dummies(x)
    def Linear_Regression(self, x_train, y_train, x_test, y_test):
        reg = LinearRegression().fit(x_train,y_train)
        predictions = reg.predict(x_test)
        predictions_train = reg.predict(x_train)
        error_metrics = {}
        error_metrics['MSE_test'] = mean_squared_error(y_test, predictions)
        error_metrics['MSE_train'] = mean_squared_error(y_train, predictions_train)
        return st.markdown('### MSE Train: ' + str(round(error_metrics['MSE_train'], 3)) +
                           ' -- MSE Test: ' + str(round(error_metrics['MSE_test'], 3)))




    def Logistic_Regression(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', LogisticRegression())])
        pipelines = [pipeline_dt]
        best_accuracy = 0.0
        best_classifier = 0
        best_pipeline = ""
        pipe_dict = {0: 'Decision Tree'}
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return (classification_report(y_test, model.predict(x_test)))

    def Decision_Tree(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', DecisionTreeClassifier())])
        pipelines = [pipeline_dt]
        best_accuracy = 0.0
        best_classifier = 0
        best_pipeline = ""
        pipe_dict = {0: 'Decision Tree'}
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return (classification_report(y_test, model.predict(x_test)))

    def RandomForest(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', RandomForestClassifier())])
        pipelines = [pipeline_dt]
        best_accuracy = 0.0
        best_classifier = 0
        best_pipeline = ""
        pipe_dict = {0: 'Decision Tree'}
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return (classification_report(y_test, model.predict(x_test)))

    def naive_bayes(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', GaussianNB())])
        pipelines = [pipeline_dt]
        best_accuracy = 0.0
        best_classifier = 0
        best_pipeline = ""
        pipe_dict = {0: 'Decision Tree'}
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return (classification_report(y_test, model.predict(x_test)))

    def knearest(self, x_train, y_train, x_test, y_test):
        pipeline_dt = Pipeline([('dt_classifier', KNeighborsClassifier())])
        pipelines = [pipeline_dt]
        best_accuracy = 0.0
        best_classifier = 0
        best_pipeline = ""
        pipe_dict = {0: 'K Nearest Neighbour'}
        for pipe in pipelines:
            pipe.fit(x_train, y_train)
        for i, model in enumerate(pipelines):
            return (classification_report(y_test, model.predict(x_test)))

    # def XGb_classifier(self, x_train, y_train, x_test, y_test):
    #     pipeline_dt = Pipeline([('dt_classifier', XGBClassifier())])
    #     pipelines = [pipeline_dt]
    #     best_accuracy = 0.0
    #     best_classifier = 0
    #     best_pipeline = ""
    #     pipe_dict = {0: 'Decision Tree'}
    #     for pipe in pipelines:
    #         pipe.fit(x_train, y_train)
    #     for i, model in enumerate(pipelines):
    #         return (classification_report(y_test, model.predict(x_test)))
