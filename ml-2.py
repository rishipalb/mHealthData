import streamlit as st
import pandas as pd
import seaborn as sns
import altair as alt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

st.title("Machine Learning App")
st.write("Upload a CSV file and select a machine learning technique to apply")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

ml_type = st.sidebar.selectbox("Select a machine learning technique", ["None", "K-Means Clustering", "Random Forest Classification", "Multiple Regression"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("Data preview:")
    st.write(data.head())

# If K-Means clustering is selected, display a slider for selecting the number of clusters
    if ml_type == "K-Means Clustering":
        st.write("Select the number of clusters:")
        n_clusters = st.slider("Number of clusters", min_value=2, max_value=10)

        # Apply K-Means clustering to the data
        kmeans = KMeans(n_clusters=n_clusters, n_init="auto")
        kmeans.fit(data)
        groups = kmeans.labels_

        # Display the clustering results in a scatter plot
        pca = PCA(n_components=2)
        principal_components = pca.fit_transform(data)
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
 # If Random Forest classification is selected, display a slider for selecting the test size
    elif ml_type == "Random Forest Classification":
        st.write("Select the test size:")
        test_size = st.slider("Test size", min_value=0.1, max_value=0.5, step=0.1)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(data.drop('label', axis=1), data['label'], test_size=test_size)

        # Train a Random Forest classifier on the training set
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)

        # Evaluate the classifier on the testing set and display the results
        y_pred = clf.predict(X_test)
        st.write(classification_report(y_test, y_pred))

  # If Multiple Regression is selected, display a list of independent variables and a dependent variable
    elif ml_type == "Multiple Regression":
        st.write("Select the dependent variable:")
        target_var = st.selectbox("Target Variable", list(data.columns))

        st.write("Select the independent variables:")
        independent_vars = st.multiselect("Independent Variables", list(data.columns), default='ekg_1')

        # Train a Multiple Regression model and display the results
        X = data[independent_vars]
        y = data[target_var]
        reg = LinearRegression().fit(X, y)
        st.write("R-squared:", reg.score(X, y))

