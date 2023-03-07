import pandas as pd
import numpy as np
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt
import seaborn as sns 
import scipy.stats as stats


def main():
    st.title('Introduction to building Streamlit WebApp')
    st.sidebar.title('This is the sidebar')
    st.sidebar.markdown('Let’s start with binary classification!!')

#def diagnostic_plots(df, variable):
    # function to plot a histogram and a Q-Q plot
    # side by side, for a certain variable
    
#    plt.figure(figsize=(15,6))
#    plt.subplot(1, 2, 1)
#    df[variable].hist()

#    plt.subplot(1, 2, 2)
#    stats.probplot(df[variable], dist="norm", plot=plt)

#    #plt.show()
#    st.pyplot()


if __name__ == "__main__":
    main()

@st.cache_data(persist= True)
def load():
    data= pd.read_csv("data/Updated_Subset_time_1.csv")
#    label= LabelEncoder()
#    for i in data.columns:
#        data[i] = label.fit_transform(data[i])
    return data
df = load()
### Convert entire 'timeColumn' to timedelta type..
#df['time_sec'] = pd.to_timedelta(df['time_sec'])
# Convert to integer value:
#df["time_sec"] = df["time_sec"].dt.seconds

### Convert 'timeColumn' to minutes only.
#df['columnAsMinutes'] = df['time_sec'].dt.total_seconds()
### Convert 'timeColumn' to seconds.
#df['columnAsSeconds'] = df['time_sec'].dt.total_seconds()

#df['time_sec'] = pd.Timedelta(Second(df['sec']))
def_df = df.drop(columns=["sub_id", "time_sec", "label"])

dfd = def_df.describe(include='all')
st.write('Dataframe Description')
st.write(dfd)

act_df = df.drop(columns=["sub_id"])

y_val= st.sidebar.selectbox("Pick a variable", list(act_df.columns))

# Plot for time series
chart = alt.Chart(act_df).mark_line().encode(
    x='time_sec:Q',
    y=f'{y_val}',
    #color='label'
).properties(
    width=300,
    height=300
).facet(
    facet='label',
    columns=2
)
st.write('Time series plots')
st.write(chart)

# Box plot for IQR and outliers
box = alt.Chart(act_df).mark_boxplot(size=50, extent=0.5, outliers={'size': 5}).encode(
    x='label:O',
    y=f'{y_val}'
).properties(
    width=700,
    height=300
)
st.write('Box-plot')
st.write(box)

# Histogram for distribution
hist = alt.Chart(act_df).mark_bar().encode(
    alt.X(f'{y_val}', bin=True),
    y='count()',
).properties(
    width=300,
    height=300
).facet(
    facet='label',
    columns=2
)
st.write('Histogram')
st.write(hist)




# Transformation of the variables using log, reciprocal, square root and exponential method
tran_var= st.selectbox("Pick a transformation", ('Log', 'Reciprocal', 'Square root', 'Exponential'))

# Log transformation
if tran_var == "Log":
    act_df['Log_var']=np.log(act_df[f'{y_val}']+1)
    hist_log = alt.Chart(act_df).mark_bar().encode(
    alt.X('Log_var', bin=True),
    y='count()',
    ).properties(
        width=300,
        height=300
    ).facet(
        facet='label',
        columns=2
    )
    st.write('Histogram for lograthmic transformation')
    st.write(hist_log)

# Reciprocal transformation
if tran_var == "Reciprocal":
    act_df['Rec_var']=1/(act_df[f'{y_val}']+1)
    hist_rec = alt.Chart(act_df).mark_bar().encode(
    alt.X('Rec_var', bin=True),
    y='count()',
    ).properties(
        width=300,
        height=300
    ).facet(
        facet='label',
        columns=2
    )
    st.write('Histogram for reciprocal tranformation')
    st.write(hist_rec)

# Square root transformation
if tran_var == "Square root":
    act_df['Sqr_var']=act_df[f'{y_val}']**(1/2)
    
    hist_sqr = alt.Chart(act_df).mark_bar().encode(
    alt.X('Sqr_var', bin=True),
    y='count()',
    ).properties(
        width=300,
        height=300
    ).facet(
        facet='label',
        columns=2
    )
    st.write('Histogram for square root transformation')
    st.write(hist_sqr)


# Exponential transformation
if tran_var == "Exponential":
    act_df['Exp_var']=act_df[f'{y_val}']**(1/5)
    hist_exp = alt.Chart(act_df).mark_bar().encode(
    alt.X('Exp_var', bin=True),
    y='count()',
    ).properties(
        width=300,
        height=300
    ).facet(
        facet='label',
        columns=2
    )
    st.write('Histogram for exponential transformation')
    st.write(hist_exp)


#st.write(sns.boxplot(x=f'{y_val}', y='label', palette="husl", data=act_df))
#st.pyplot()
load.clear()
