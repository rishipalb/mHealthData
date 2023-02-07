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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, precision_recall_curve
from sklearn.metrics import precision_score, recall_score

def main():
    st.title('Introduction to building Streamlit WebApp')
    st.sidebar.title('This is the sidebar')
    st.sidebar.markdown('Let’s start with binary classification!!')
if __name__ == "__main__":
    main()

@st.cache(persist= True)
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


@st.cache(persist=True)
def split(df):
    y = df.label
    x = df.drop(columns=["label"])
    x_train, x_test, y_train, y_test =     train_test_split(x,y,test_size=0.3, random_state=0)
    
    return x_train, x_test, y_train, y_test
x_train, x_test, y_train, y_test = split(df)

def plot_metrics(metrics_list):
    if "Confusion Matrix" in metrics_list:
        st.subheader("Confusion Matrix")
        ConfusionMatrixDisplay(model, x_test, y_test, display_labels= class_names)
        st.pyplot()
    if "ROC Curve" in metrics_list:
        st.subheader("ROC Curve")
        RocCurveDisplay(model, x_test, y_test)
        st.pyplot()
    if "Precision-Recall Curve" in metrics_list:
        st.subheader("Precision-Recall Curve")
        precision_recall_curve(model, x_test, y_test)
        st.pyplot()
class_names = ["walk", "run"]

st.sidebar.subheader("Choose classifier")
classifier = st.sidebar.selectbox("Classifier", ("Support Vector Machine (SVM)", "Logistic Regression", "Random Forest"))

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

