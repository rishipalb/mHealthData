import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler 
from sklearn import datasets 
import seaborn as sns
import streamlit as st

df = pd.read_csv('data/Updated_Subset_1.csv')
df.drop('sub_id', axis = 1, inplace = True)
col_names = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ekg_1', 'ekg_2', 'acc_ankel_x', 'acc_ankle_y', 'acc_ankle_z', 'gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z', 'mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z', 'acc_arm_x', 'acc_arm_y', 'acc_arm_z', 'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z', 'mag_arm_x', 'mag_arm_y', 'mag_arm_z', 'label']
df.columns = col_names
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X = X.values
y = y.values

cols = ['acc_chest_x', 'acc_chest_y', 'acc_chest_z', 'ekg_1', 'ekg_2', 'acc_ankel_x', 'acc_ankle_y', 'acc_ankle_z', 'gyro_ankle_x', 'gyro_ankle_y', 'gyro_ankle_z', 'mag_ankle_x', 'mag_ankle_y', 'mag_ankle_z', 'acc_arm_x', 'acc_arm_y', 'acc_arm_z', 'gyro_arm_x', 'gyro_arm_y', 'gyro_arm_z', 'mag_arm_x', 'mag_arm_y', 'mag_arm_z']

#cols = list(df.columns)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()
classifier.fit(X_train, y_train)

X_train = pd.DataFrame(X_train, columns=[cols])
X_test = pd.DataFrame(X_test, columns=[cols])

import numpy as np 
importances = classifier.feature_importances_ 
# # Sort the feature importance in descending order # 
sorted_indices = np.argsort(importances)[::-1] 
feat_labels = df.columns[1:] 
#for f in range(X_train.shape[1]): 
#    st.write("%2d) %-*s %f" % (f + 1, 30, feat_labels[sorted_indices[f]], importances[sorted_indices[f]]))

# predicting on the test dataset
y_pred = classifier.predict(X_test)

# finding out the accuracy
from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
st.markdown("### Champion model selected: Random Forest")
st.write("Accuracy:", score)


import matplotlib.pyplot as plt 
plt.title('Feature Importance') 
plt.bar(range(X_train.shape[1]), importances[sorted_indices], align='center') 
plt.xticks(range(X_train.shape[1]), X_train.columns[sorted_indices], rotation=90) 
plt.tight_layout() 
#plt.show()
st.pyplot()

# pickling the model
import pickle
pickle_out = open("classifier.pkl", "wb")
pickle.dump(classifier, pickle_out)
pickle_out.close()
