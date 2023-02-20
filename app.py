import pandas as pd

import ydata_profiling

import streamlit as st

from streamlit_pandas_profiling import st_profile_report

from pandas_profiling import ProfileReport



st.sidebar.header("Pick an activity for profiling")
x_val= st.sidebar.selectbox("Pick a type of activity", ('Walking (1 min)', 'Running (1 min)'))


#df = pd.read_csv("crops data.csv", na_values=['='])
df = pd.read_csv('data/Updated_Subset_1.csv', na_values=['='])

act_val = 0

if x_val == 'Walking (1 min)':
    act_val = 0
if x_val == 'Running (1 min)':
    act_val = 1



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
if st.sidebar.checkbox("Display Statistics", False):
    st.subheader("Show mHealth Profile Statistics")
    st_profile_report(profile)