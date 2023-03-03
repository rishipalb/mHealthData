from numpy.core.numeric import True_
from sklearn import metrics
import streamlit as st
import pandas as pd
import numpy as np
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

st.set_option('deprecation.showPyplotGlobalUse', False)
def main():
    st.title('Introduction to building Streamlit WebApp')
    st.sidebar.title('This is the sidebar')
    st.sidebar.markdown('Let’s start with binary classification!!')
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

#df.loc[df.label != '4', 'label'] = 0
#df.loc[df.label == '4', 'label'] = 1

if st.sidebar.checkbox("Display data", False):
    st.subheader("Show mHealth dataset")
    st.write(df)


@st.cache_data(persist=True)
def split(df):
    y = df.label
    x = df.drop(columns=["sub_id", "label"])
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

st.sidebar.subheader("Select a machine learning model")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest", "Decision Tree", "K-Means Clustering","Multiple Regression"))
if classifier == "Support Vector Machine (SVM)":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C")
    kernel = st.sidebar.radio("Kernel", ("rbf", "linear"), key="kernel") 
    gamma = st.sidebar.radio("Gamma (Kernal coefficient", ("scale", "auto"), key="gamma")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))

    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Support Vector Machine (SVM) results")
        model = SVC(C=C, kernel=kernel, gamma=gamma)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2)) 
        plot_metrics(metrics)

if classifier == "Logistic Regression":
    st.sidebar.subheader("Hyperparameters")
    C = st.sidebar.number_input("C (Regularization parameter)", 0.01, 10.0, step=0.01, key="C_LR")
    max_iter = st.sidebar.slider("Maximum iterations", 100, 500, key="max_iter")
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Logistic Regression Results")
        model = LogisticRegression(C=C, max_iter=max_iter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

if classifier == "Random Forest":
    st.sidebar.subheader("Hyperparameters")
    n_estimators= st.sidebar.number_input("The number of trees in the forest", 100, 5000, step=10, key="n_estimators")
    max_depth = st.sidebar.number_input("The maximum depth of tree", 1, 20, step =1, key="max_depth")
    bootstrap = st.sidebar.radio("Bootstrap samples when building trees", options=[True, False], key="bootstrap")
    
    metrics = st.sidebar.multiselect("What metrics to plot?", ("Confusion Matrix", "ROC Curve", "Precision-Recall Curve"))
    
    if st.sidebar.button("Classify", key="classify"):
        st.subheader("Random Forest Results")
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, bootstrap= bootstrap, n_jobs=-1 )
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write("Accuracy: ", accuracy.round(2))
        st.write("Precision: ", precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write("Recall: ", recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

if classifier == 'Decision Tree':
    st.sidebar.subheader('Model parameters')
    #choose parameters
 
    criterion= st.sidebar.radio('Criterion(measures the quality of split)', ('gini', 'entropy'), key='criterion')
    splitter = st.sidebar.radio('Splitter (How to split at each node?)', ('best', 'random'), key='splitter')
 
    metrics = st.sidebar.multiselect('Select your metrics : ', ('Confusion Matrix', 'ROC Curve', 'Precision-Recall Curve'))
 
    if st.sidebar.button('Classify', key='classify'):
        st.subheader('Decision Tree Results')
        model = DecisionTreeClassifier(criterion=criterion, splitter=splitter)
        model.fit(x_train, y_train)
        accuracy = model.score(x_test, y_test)
        y_pred = model.predict(x_test)
        st.write('Accuracy: ', accuracy.round(2)*100,'%')
        st.write('Precision: ', precision_score(y_test, y_pred, labels=class_names).round(2))
        st.write('Recall: ', recall_score(y_test, y_pred, labels=class_names).round(2))
        plot_metrics(metrics)

# If K-Means clustering is selected, display a slider for selecting the number of clusters
if classifier == "K-Means Clustering":
        
            #data_k= pd.read_csv("data/Updated_Subset_1.csv")
            data_k = df.drop(columns=["sub_id", "label"])
            st.write("Select the number of clusters:")
            n_clusters = st.slider("Number of clusters", min_value=2, max_value=10)

            # Apply K-Means clustering to the data
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(data_k)
            labels = kmeans.labels_

            # Display the clustering results in a scatter plot
            pca = PCA(n_components=2)
            principal_components = pca.fit_transform(data_k)
            principal_df = pd.DataFrame(data=principal_components, columns=['PC1', 'PC2'])
            principal_df['label'] = labels
            st.write(sns.scatterplot(data=principal_df, x='PC1', y='PC2', hue='label'))

  # If Multiple Regression is selected, display a list of independent variables and a dependent variable
if classifier == "Multiple Regression":
        #data_reg = pd.read_csv("data/Updated_Subset_1.csv")
        data_reg = df.drop(columns=["sub_id", "label"])
        st.write("Select the dependent variable:")
        target_var = st.selectbox("Target Variable", list(data_reg.columns))

        st.write("Select the independent variables:")
        independent_vars = st.multiselect("Independent Variables", list(data_reg.columns), default=['ekg_1'])

        # Train a Multiple Regression model and display the results
        X = data_reg[independent_vars]
        y = data_reg[target_var]
        reg = LinearRegression().fit(X, y)
        st.write("R-squared:", reg.score(X, y))

load.clear()
split.clear()