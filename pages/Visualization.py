import pandas as pd
import streamlit as st
import altair as alt
from matplotlib import pyplot as plt
import seaborn as sns 

def main():
    st.title('Introduction to building Streamlit WebApp')
    st.sidebar.title('This is the sidebar')
    st.sidebar.markdown('Let’s start with binary classification!!')
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

box = alt.Chart(act_df).mark_boxplot(extent='min-max').encode(
    x='label:O',
    y=f'{y_val}'
).properties(
    width=700,
    height=300
)
st.write('Box-plot')
st.write(box)

#st.write(sns.boxplot(x=f'{y_val}', y='label', palette="husl", data=act_df))
#st.pyplot()
load.clear()
