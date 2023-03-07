# mHealthData
For this project, we used the mHealth dataset which consists of twelve activities. As a proof of concept, we used the activities such as walking and running to build a predictive model for differentiating these activities.


a.     Data Source(s): 
The dataset for this assignment has data on measurements of body motion and vital signs collected from 10 volunteers performing 12 physical activities.  The sensors captured the activities at a sampling rate of 50Hz. For the purpose of this project only two activities were selected, namely, Walking (1min) and Running (1min). The data dictionary is given below:

Activity set

The activity set is listed in the following:
L1: Standing still (1 min) 
L2: Sitting and relaxing (1 min) 
L3: Lying down (1 min) 
L4: Walking (1 min) 
L5: Climbing stairs (1 min) 
L6: Waist bends forward (20x) 
L7: Frontal elevation of arms (20x)
L8: Knees bending (crouching) (20x)
L9: Cycling (1 min)
L10: Jogging (1 min)
L11: Running (1 min)
L12: Jump front & back (20x)

NOTE: In brackets are the number of repetitions (Nx) or the duration of the exercises (min).

Dataset files
The data collected for each subject is stored in a different log file: 'mHealth_subject<SUBJECT_ID>.log'.
Each file contains the samples (by rows) recorded for all sensors (by columns).
The labels used to identify the activities are similar to the ones presented in Section 2 (e.g., the label for walking is '4'). The dataset is available at: https://github.com/rishipalb/mHealthData/blob/main/data/Updated_Subset_time_1.csv

The meaning of each column is detailed next:

Table 1. Description of the dataset
Column No.
Column ID
Description
1. sub_id: Subject ID
2. acc_chest_x: acceleration from the chest sensor (X axis)
3. acc_chest_y:acceleration from the chest sensor (Y axis)
4. acc_chest_z: acceleration from the chest sensor (Z axis)
5. ekg_1: electrocardiogram signal (lead 1)
6. ekg_2: electrocardiogram signal (lead 2)
7. acc_ankle_x: acceleration from the left-ankle sensor (X axis)
8. acc_ankle_y: acceleration from the left-ankle sensor (Y axis)
9. acc_ankle_z: acceleration from the left-ankle sensor (Z axis)
10. gyro_ankle_x: gyro from the left-ankle sensor (X axis)
11. gyro_ankle_y: gyro from the left-ankle sensor (Y axis)
12. gyro_ankle_z: gyro from the left-ankle sensor (Z axis)
13. mag_ankle_x: magnetometer from the left-ankle sensor (X axis)
14. mag_ankle_y: magnetometer from the left-ankle sensor (Y axis)
15. mag_ankle_z: magnetometer from the left-ankle sensor (Z axis)
16. acc_arm_x: acceleration from the right-lower-arm sensor (X axis)
17. acc_arm_y: acceleration from the right-lower-arm sensor (Y axis)
18. acc_arm_z: acceleration from the right-lower-arm sensor (Z axis)
19. gyro_arm_x: gyro from the right-lower-arm sensor (X axis)
20. gyro_arm_y: gyro from the right-lower-arm sensor (Y axis)
21. gyro_arm_z: gyro from the right-lower-arm sensor (Z axis)
22. mag_arm_x: magnetometer from the right-lower-arm sensor (X axis)
23. mag_arm_y: magnetometer from the right-lower-arm sensor (Y axis)
24. mag_arm_z: magnetometer from the right-lower-arm sensor (Z axis)
25. time_sec: Time in seconds
26. label: Label (0 for walking and 1 for running)



*Units: Acceleration (m/s^2), gyroscope (deg/s), magnetic field (local), ecg (mV)

The original dataset is located at: https://archive.ics.uci.edu/ml/datasets/MHEALTH+Dataset

For the subset dataset used for analysis for this project, the column ‘Label’ designates activity: Walking (1min) as “0” and Running (1min) as “1”.

Streamlit app is available at: https://rishipalb-mhealthdata-app-s7u1np.streamlit.app/

