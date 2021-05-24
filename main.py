import pandas as pd
import streamlit as st
from AttributeInformation import *
from EDADataFrameAnalysis import *
from DataframeLoader import *
from DataBaseModelling import *

st.set_option('deprecation.showPyplotGlobalUse', False)

# image = Image.open('cover.jpg')
matplotlib.use("Agg")


def main():
    st.header("Machine Learning Application for Automated EDA")
    activities = ["General EDA", "Different Models", "About Us"]
    st.sidebar.header("Select Activities")
    choice = st.sidebar.radio("Go to", activities)

    if choice == 'General EDA':
        st.subheader("Exploratory Data Analysis")
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")
            if st.checkbox("Show dtypes"):
                st.write(dataframe.show_dtypes(df))

            if st.checkbox("Show Columns"):
                st.write(dataframe.show_columns(df))

            if st.checkbox("Show Missing"):
                st.write(dataframe.Show_Missing1(df))

            if st.checkbox("column information"):
                st.write(info.Column_information(df))

            if st.checkbox("Aggregation Tabulation"):
                st.write(dataframe.Tabulation(df))

            if st.checkbox("Num Count Summary"):
                st.write(info.num_count_summary(df))

            if st.checkbox("Statistical Summary"):
                st.write(info.statistical_summary(df))
            if st.checkbox("Show Selected Columns"):
                selected_columns = st.multiselect("Select Columns", dataframe.show_columns(df))
                new_df = df[selected_columns]
                st.dataframe(new_df)

            if st.checkbox("Numerical Variables"):
                num_df = dataframe.Numerical_variables(df)
                numer_df = pd.DataFrame(num_df)
                st.dataframe(numer_df)

            if st.checkbox("Categorical Variables"):
                new_df = dataframe.categorical_variables(df)
                catego_df = pd.DataFrame(new_df)
                st.dataframe(catego_df)
                all_columns_names = dataframe.show_columns(df)
                all_columns_names1 = dataframe.show_columns(df)
                selected_columns_names = st.selectbox("Select Column 1 For Cross Tabultion", all_columns_names)
                selected_columns_names1 = st.selectbox("Select Column 2 For Cross Tabultion", all_columns_names1)
                if st.button("Generate Cross Tab"):
                    st.dataframe(pd.crosstab(df[selected_columns_names], df[selected_columns_names1]))

                all_columns_names3 = dataframe.show_columns(df)
                all_columns_names4 = dataframe.show_columns(df)
                selected_columns_name3 = st.selectbox("Select Column 1 For Pearsonr Correlation (Numerical Columns)",
                                                      all_columns_names3)
                selected_columns_names4 = st.selectbox("Select Column 2 For Pearsonr Correlation (Numerical Columns)",
                                                       all_columns_names4)

                #spearmanr3 = dataframe.show_columns(df)
                spearmanr4 = dataframe.show_columns(df)
                spearmanr13 = st.selectbox("Select Column 1 For spearmanr Correlation (Categorical Columns)",
                                           spearmanr4)
                spearmanr14 = st.selectbox("Select Column 2 For spearmanr Correlation (Categorical Columns)",
                                           spearmanr4)
                if st.button("Generate spearmanr Correlation"):
                    df = pd.DataFrame(dataframe.Show_spearmanr(catego_df[spearmanr13], catego_df[spearmanr14]),
                                      index=['Pvalue', '0'])
                    st.dataframe(df)

                st.subheader("UNIVARIATE ANALYSIS")

                all_columns_names = dataframe.show_columns(df)
                selected_columns_names = st.selectbox("Select Column for Histogram ", all_columns_names)
                if st.checkbox("Show Histogram for Selected variable"):
                    st.write(dataframe.show_hist(df[selected_columns_names]))
                    st.pyplot()

                all_columns_names = dataframe.show_columns(df)
                selected_columns_names = st.selectbox("Select Columns Distplot ", all_columns_names)
                if st.checkbox("Show DisPlot for Selected variable"):
                    st.write(dataframe.Show_DisPlot(df[selected_columns_names]))
                    st.pyplot()

                all_columns_names = dataframe.show_columns(df)
                selected_columns_names = st.selectbox("Select Columns CountPlot ", all_columns_names)
                if st.checkbox("Show CountPlot for Selected variable"):
                    st.write(dataframe.Show_CountPlot(df[selected_columns_names]))
                    st.pyplot()
                st.subheader("MULTIVARIATE ANALYSIS")
                if st.checkbox("Show Histogram"):
                    st.write(dataframe.show_hist(df))
                    st.pyplot()
                if st.checkbox("Show HeatMap"):
                    st.write(dataframe.Show_HeatMap(df))
                    st.pyplot()

                if st.checkbox("Show PairPlot"):
                    st.write(dataframe.Show_PairPlot(df))
                    st.pyplot()

    elif choice == 'About Us':
        html_temp2 = """
        <div style="padding:10px;color:#fff;">
            <h3>Overview</h3>
            <p>Our product is we want to automate how people do Data Analysis. We want everyone to access the power 
               of Data Science and Data Analytics .Even non technical people can also use the tool with ease.
               The UI is also very easy to understand and also is very fast as compared to the current system.
               We also want the user should be able use different stats and maths techniques with ease.
            </p>
            <h3>Features</h3>
            <ul>
                <li>User will be able to do Deal with the Missing values by using the variety of statistical methods available.</li>
                <li>User can also do EDA(Exploratory Data Analytics) and get visualizations done using this tool.</li>
                <li>User will also be able to have different algorithms to apply to the given dataset.</li>
                <li>User can also analyze how different machine learning fit the dataset.</li>
                <li>User can also view what datasets he/she has worked on.</li>
                <li>User can also do all the data analysis without even writing a single line of code.</li>
            </ul>
        </div>
        """
        st.markdown(html_temp2, unsafe_allow_html=True)

    elif choice == 'Different Models':
        data = st.file_uploader("Upload a Dataset", type=["csv"])
        html_temp2 = """<div style="background-color:#000000;padding:10px">
                                               <h1 style="color:white;text-align:center;">Different Models </h1>
                                               </div>
                                       <div>
                                       </br>"""
        st.markdown(html_temp2, unsafe_allow_html=True)
        if data is not None:
            df = load.read_csv(data)
            st.dataframe(df.head())
            st.success("Data Frame Loaded successfully")
            df1 = df
            cols = list(df1.columns.values)
            st.write(df.isnull().sum())
            fill = st.selectbox('Imputer Or Drop Column:', options=['Select', 'Impute Knn', 'DropNA'])

            if fill == 'Impute Knn':
                df = dataframe.imputee(df)
                st.subheader('Missing after Impute Knn')
                st.write(df.isna().sum())
            elif fill == "DropNA":
                imp_df = dataframe.impute(df)
                st.dataframe(imp_df)
                df.dropna(inplace=True)
                st.subheader('Missing After DropNA')
                st.write(df.isna().sum())

            if st.checkbox("Categorical Variables"):
                new_df = dataframe.categorical_variables(df)
                catego_df = pd.DataFrame(new_df)
                categories = list(catego_df.columns.values)
                st.dataframe(catego_df)
                if st.checkbox('Encode Categorical Features'):
                    dummies = pd.get_dummies(df[categories])
                    st.dataframe(dummies)
                    df = df.drop(categories, axis=1)
                    df = df.join(dummies)
                    # st.dataframe(df)

            col = list(df.columns.values)
            st.header("Select Columns for Model Building")
            if st.checkbox("Select your Variables  (Target Variable should be at last)"):
                selected_columns_ = st.multiselect("Select Columns for seperation ", options=col, default=col)
                sep_df = df[selected_columns_]
                # st.dataframe(sep_df)

                st.subheader("Indpendent Data")
                try:
                    x = sep_df.iloc[:, :-1]
                    st.write(x)
                except IndexError:
                    st.write('Please Select Some Columns ')
                st.subheader("Show Dependent Data")
                try:
                    y = sep_df.iloc[:, -1]
                    st.write(y)
                    from sklearn.model_selection import train_test_split

                    x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

                except IndexError:
                    st.write('Please Select Some Columns')

            st.subheader("MODEL BUILDING")
            st.write("Build your BaseLine Model")

    type = st.sidebar.selectbox("Algorithm type", ("Classification", "Regression", "Clustering"))

    if type == "Regression":
        chosen_classifier = st.sidebar.selectbox("Please choose a classifier",
                                                 ('Random Forest', 'Linear Regression', 'Neural Network'))
    elif type == "Classification":
        chosen_classifier = st.sidebar.selectbox("Please choose a classifier",
                                                 (
                                                 'Logistic Regression', 'Naive Bayes', 'Random Forest', 'Decision Tree',
                                                 'K nearest Neighbour'))

    if st.sidebar.button('Confirm Model'):
        st.write(type, chosen_classifier)
        if type == 'Classification' and chosen_classifier == 'Logistic Regression':
            x = model.Logistic_Regression(x_train, y_train, x_test, y_test)
            st.header('Logistic Regression')
            st.write(x)
        elif type == 'Classification' and chosen_classifier == 'Naive Bayes':
            x = model.naive_bayes(x_train, y_train, x_test, y_test)
            st.header('Naive Bayes Classifier')
            st.write(x)
        elif type == 'Classification' and chosen_classifier == 'K nearest Neighbour':
            x = model.knearest(x_train, y_train, x_test, y_test)
            st.header('K Nearest Neighbour')
            st.write(x)
        elif type == 'Classification' and chosen_classifier == 'Random Forest':
            x = model.RandomForest(x_train, y_train, x_test, y_test)
            st.header('Random Forest')
            st.write(x)
        elif type == 'Classification' and chosen_classifier == 'Decision Tree':
            x = model.Decision_Tree(x_train, y_train, x_test, y_test)
            st.header('Decision Tree')
            st.write(x)
        elif type == 'Regression' and chosen_classifier == 'Linear Regression':
            x = model.Linear_Regression(x_train, y_train, x_test, y_test)
            st.header('Linear Regression')
            # st.write(x)



st.markdown('<style>h1{color: red;}</style>', unsafe_allow_html=True)
# st.title("Credits and Inspiration")
# """https://pycaret.org/"""

if __name__ == '__main__':
    load = DataFrame_Loader()
    dataframe = EDA_Dataframe_Analysis()
    info = Attribute_Information()
    model = Data_Base_Modelling()
    main()
