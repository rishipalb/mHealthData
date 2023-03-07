import streamlit as st
import pandas as pd
from func import FeatureSelector
from charts import *
from models import ModelRunner
import warnings
from sklearn.exceptions import DataConversionWarning

import altair as alt
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.metrics import precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.metrics import classification_report

warnings.filterwarnings(action='ignore', category=DataConversionWarning)

df = None # global dataframe; if it is not uploaded, it is None.
st.title('Feature Selection')
st.sidebar.title('This is the sidebar')
st.sidebar.markdown('Choose an option!!')

def main():
    global df
    #Title
#def main():
   
if __name__ == "__main__":
    main()

@st.cache_data(persist= True)
def load():
    data= pd.read_csv("data/Updated_Subset_1.csv")
#    label= LabelEncoder()
#    for i in data.columns:
#        data[i] = label.fit_transform(data[i])
    return data
df = load()

df = df.drop(columns=["sub_id"])


@st.cache_data(persist=True)
def split(df):
    y = df.label
    x = df.drop(columns=["label"])
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=0)
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay.from_estimator(model, x_test, y_test, display_labels=class_names)
        st.pyplot()
        #cm=confusion_matrix(y_test, y_pred)
        #ConfusionMatrixDisplay(cm,model.classes_).plot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        PrecisionRecallDisplay.from_estimator(model, x_test, y_test)
        st.pyplot()
class_names = ["walk", "run"]

tab1, tab2, tab3 = st.tabs(['Feature Selection', 'Dimension Reduction', 'Regression'])
with tab1:
        if st.sidebar.checkbox("Display data", False):
            st.subheader("Show mHealth dataset")
            st.write(df)
        if df is not None:
            #display dataframe
            #st.write(df)
            #select target variable
            target = st.selectbox("Select Target Feature",df.columns)
            #select feature selection method
            selector = st.radio(label="Selection Method",options=["SelectKBest","RFE","SelectFromModel"])
            F = FeatureSelector(df,target)
            univariate,ref,sfm,problem = F.get_result_dictionaries()
            #chart
            if selector == "SelectKBest":
                fig = barchart(univariate["scores"], univariate["feature_names"], "Feature Scores acc to SelectKBest")
            elif selector == "RFE":
                fig = barchart(ref["ranking"], ref["feature_names"], "Ranking acc to RFE; (Lower better)")
            elif selector == "SelectFromModel":
                fig = barchart(sfm["scores"], sfm["feature_names"], "Feature Scores acc to SelectFromModel")
            st.pyplot(fig)
            #select k number of features to proceed
            k = st.number_input("Number of Feature to proceed (k): ", min_value=0, max_value= len(df.columns) - 1)
            if problem == "regression":
                model = st.selectbox("ML Method",["Linear Regression","XGBoost"])
            else:
                model = st.selectbox("ML Method",["Logistic Regression","Decision Tree"])
            #when k is determined 
            if k > 0:
                #get last X,y according to feature selection
                X,_,temp,col_types,_ = F.extract_x_y() 
                y = df[target].values.reshape(-1,1)
                #feature set
                if selector == "SelectKBest":
                    X = F.univariate_feature_selection(X,y,temp,k)["X"]
                elif selector == "RFE":
                    X = F.ref_feature_selection(X,y,temp,col_types,k)["X"]
                elif selector == "SelectFromModel":
                    X = F.sfm_feature_selection(X,y,temp,col_types,k)["X"]
                #run models
                M = ModelRunner(model,X,y,problem)
                score = M.runner()
                #display score
                st.write("Score of Model: {}".format(score))

with tab2:

    st.subheader("Select a model")
    model = st.selectbox("Model", ("none", "K-Means Clustering"))
    st.write(f"Model: {model}")

    # If K-Means clustering is selected, display a slider for selecting the number of clusters
    if model == "K-Means Clustering":
        
            #data_k= pd.read_csv("data/Updated_Subset_1.csv")
            data_k = df.drop(columns=["label"])
            st.write("Select the number of clusters:")
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10)

            # Apply K-Means clustering to the data
            kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
            kmeans.fit(data_k)
            groups = kmeans.labels_

            # Display the clustering results in a scatter plot
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(data_k)
            principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            principal_df['group'] = groups
            # st.write(sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='label'))
            # scatter = sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='group')
            # st.write(scatter)
            chart = alt.Chart(principal_df).mark_circle(size=60).encode(
            x='PC1:Q',
            y='PC2:Q',
            color='group'
            ).properties(
            width=600,
            height=600
            )
            st.write('K-Mean Cluster')
            st.write(chart)

with tab3:
        #data_reg = pd.read_csv("data/Updated_Subset_1.csv")
        data_reg = df.drop(columns=["label"])
        st.write("Select the dependent variable:")
        target_var = st.selectbox("Target Variable", list(data_reg.columns))

        st.write("Select the independent variables:")
        independent_vars = st.multiselect("Independent Variables", list(data_reg.columns), default=['ekg_1'])

        # Train a Multiple Regression model and display the results
        X = data_reg[independent_vars]
        y = data_reg[target_var]
        reg = LinearRegression().fit(X, y)
        st.write("R-squared:", reg.score(X, y))


main()


