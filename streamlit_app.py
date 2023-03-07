# streamlit_app.py

import streamlit as st
import snowflake.connector
import pandas as pd
import pandas_profiling
from streamlit_pandas_profiling import st_profile_report
from pandas_profiling import ProfileReport

# Initialize connection.
# Uses st.experimental_singleton to only run once.
@st.experimental_singleton
def init_connection():
    return snowflake.connector.connect(
        **st.secrets["snowflake"], client_session_keep_alive=True
    )

conn = init_connection()

# Perform query.
# Uses st.experimental_memo to only rerun when the query changes or after 10 min.
@st.experimental_memo(ttl=600)
def run_query(query):
    with conn.cursor() as cur:
        cur.execute(query)
        return cur.fetchall()

rows = run_query("SELECT * from MHEALTH_3;")
df = pd.DataFrame(rows)
# Print results.
#for row in rows:
#    st.write(f"{row[0]} has a :{row[1]}:")

st.write(df)
st.sidebar.header("Pick an activity for profiling")
x_val= st.sidebar.selectbox("Pick a type of activity", ('Null class', 'Standing still (1 min)', 'Sitting and relaxing (1 min)', 'Lying down (1 min)', 'Walking (1 min)', 'Climbing stairs (1 min)', 'Waist bends forward (20x)', 'Frontal elevation of arms (20x)', 'Knees bending (crouching) (20x)', 'Cycling (1 min)', 'Jogging (1 min)', 'Running (1 min)', 'Jump front & back (20x)'))
act_val = 0
if x_val == 'Null class':
    act_val = 0
if x_val == 'Standing still (1 min)':
    act_val = 1
if x_val == 'Sitting and relaxing (1 min)':
    act_val = 2
if x_val == 'Lying down (1 min)':
    act_val = 3
if x_val == 'Walking (1 min)':
    act_val = 4
if x_val == 'Climbing stairs (1 min)':
    act_val = 5
if x_val == 'Waist bends forward (20x)':
    act_val = 6
if x_val == 'Frontal elevation of arms (20x)':
    act_val = 7
if x_val == 'Knees bending (crouching) (20x)':
    act_val = 8
if x_val == 'Cycling (1 min)':
    act_val = 9
if x_val == 'Jogging (1 min)':
    act_val = 10
if x_val == 'Running (1 min)':
    act_val = 11
if x_val == 'Jump front & back (20x)':
    act_val = 12


df = df[df['label'] == act_val]



profile = ProfileReport(df.loc[:,'acc_chest_x':'mag_arm_z'],

                        title="mHealth Dataset",

        dataset={

        "description": "This profiling report was generated for mHealth Dataset",

        "url": "https://archive.ics.uci.edu/ml/datasets/MHEALTH%20Dataset",

    },

    variables={

        "descriptions": {
            "sub_id": "Subject identitification number",

            "acc_chest_x": "acceleration from the chest sensor (X axis)",

            "acc_chest_y": "acceleration from the chest sensor (Y axis)",

            "acc_chest_z": "acceleration from the chest sensor (Z axis)",

            "ekg_1": "electrocardiogram signal (lead 1)",

            "ekg_2": "electrocardiogram signal (lead 2)",

            "acc_ankle_x": "acceleration from the left-ankle sensor (X axis)",

            "acc_ankle_y": "acceleration from the left-ankle sensor (Y axis)",

            "acc_ankle_z": "acceleration from the left-ankle sensor (Z axis)",

            "gyro_ankle_x": "gyro from the left-ankle sensor (X axis)",

            "gyro_ankle_y": "gyro from the left-ankle sensor (Y axis)",

            "gyro_ankle_z": "gyro from the left-ankle sensor (Z axis)",

            "mag_ankle_x": "magnetometer from the left-ankle sensor (X axis)",

            "mag_ankle_y": "magnetometer from the left-ankle sensor (Y axis)",

            "mag_ankle_z": "magnetometer from the left-ankle sensor (Z axis)",

            "acc_arm_x": "acceleration from the right-lower-arm sensor (X axis)",

            "acc_arm_y": "acceleration from the right-lower-arm sensor (Y axis)",

            "acc_arm_z": "acceleration from the right-lower-arm sensor (Z axis)",

            "gyro_arm_x": "gyro from the right-lower-arm sensor (X axis)",

            "gyro_arm_y": "gyro from the right-lower-arm sensor (Y axis)",

            "gyro_arm_z": "gyro from the right-lower-arm sensor (Z axis)",

            "mag_arm_x": "magnetometer from the right-lower-arm sensor (X axis)",

            "mag_arm_y": "magnetometer from the right-lower-arm sensor (Y axis)",

            "mag_arm_z": "magnetometer from the right-lower-arm sensor (Z axis)",

            "label": "Activity type",





        }

    }

)




st.title(f"mHealth Dataset Profiling in Streamlit for activity: {x_val}!")

st.write(df)

st_profile_report(profile)