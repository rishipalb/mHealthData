import pandas as pd
import numpy as np
import pickle
import streamlit as st
from PIL import Image

# loading in the model to predict on the data
pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)

def welcome():
	return 'welcome all'

# defining the function which will make the prediction using
# the data which the user inputs
def prediction(acc_chest_x, acc_chest_y, acc_chest_z, ekg_1, ekg_2, acc_ankle_x, acc_ankle_y, acc_ankle_z,
	       gyro_ankle_x, gyro_ankle_y, gyro_ankle_z, mag_ankle_x, mag_ankle_y, mag_ankle_z,
	        acc_arm_x, acc_arm_y, acc_arm_z, gyro_arm_x, gyro_arm_y, gyro_arm_z, mag_arm_x, mag_arm_y, mag_arm_z):

	prediction = classifier.predict(
		[[acc_chest_x, acc_chest_y, acc_chest_z, ekg_1, ekg_2, acc_ankle_x, acc_ankle_y, acc_ankle_z,
	       gyro_ankle_x, gyro_ankle_y, gyro_ankle_z, mag_ankle_x, mag_ankle_y, mag_ankle_z,
	        acc_arm_x, acc_arm_y, acc_arm_z, gyro_arm_x, gyro_arm_y, gyro_arm_z, mag_arm_x, mag_arm_y, mag_arm_z]])
	print(prediction)
	return prediction
	

# this is the main function in which we define our webpage
def main():
	# giving the webpage a title
	st.title("mHealth Dataset Prediction")
	
	# here we define some of the front end elements of the web page like
	# the font and background color, the padding and the text to be displayed
	html_temp = """
	<div style ="background-color:yellow;padding:13px">
	<h1 style ="color:black;text-align:center;">Streamlit mHealth Dataset Classifier ML App </h1>
	</div>
	"""
	
	# this line allows us to display the front end aspects we have
	# defined in the above code
	st.markdown(html_temp, unsafe_allow_html = True)

	
if __name__=='__main__':
	main()
col1, col2, col3 = st.columns(3)

with col1:
	st.write("Chest sensor measurement")
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	acc_chest_x = st.number_input("acc_chest_x", value=-6.7702)
	acc_chest_y = st.number_input("acc_chest_y", value=0.12652)
	acc_chest_z = st.number_input("acc_chest_z", value=0.77981)
	ekg_1 = st.number_input("ekg_1", value=-0.3349)
	ekg_2 = st.number_input("ekg_2", value=0.17164)

with col2:
	st.write("Ankle sensor measurement")
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	acc_ankle_x = st.number_input("acc_ankle_x", value=3.3929)
	acc_ankle_y = st.number_input("acc_ankle_y", value=-8.4832)
	acc_ankle_z = st.number_input("acc_ankle_z", value=0.1931)
	gyro_ankle_x = st.number_input("gyro_ankle_x", value=0.67718)
	gyro_ankle_y = st.number_input("gyro_ankle_y", value=-0.60788)
	gyro_ankle_z = st.number_input("gyro_ankle_z", value=0.27898)
	mag_ankle_x = st.number_input("mag_ankle_x", value=-21.672)
	mag_ankle_y = st.number_input("mag_ankle_y", value=-24.187)
	mag_ankle_z = st.number_input("mag_ankle_z", value=0.38671)

with col3:
	st.write("Arm sensor measurement")
	# the following lines create text boxes in which the user can enter
	# the data required to make the prediction
	acc_arm_x = st.number_input("acc_arm_x", value=-3.5314)
	acc_arm_y = st.number_input("acc_arm_y", value=-5.963)
	acc_arm_z = st.number_input("acc_arm_z", value=1.7019)
	gyro_arm_x = st.number_input("gyro_arm_x", value=-0.56275)
	gyro_arm_y = st.number_input("gyro_arm_y", value=-0.66324)
	gyro_arm_z = st.number_input("gyro_arm_z", value=0.72629)
	mag_arm_x = st.number_input("mag_arm_x", value=-0.85899)
	mag_arm_y = st.number_input("mag_arm_y", value=-14.086)
	mag_arm_z = st.number_input("mag_arm_z", value=24.311)

result =""
	# the below line ensures that when the button called 'Predict' is clicked,
	# the prediction function defined above is called to make the prediction
	# and store it in the variable result
if st.button("Predict"):
	result = prediction(acc_chest_x, acc_chest_y, acc_chest_z, ekg_1, ekg_2, acc_ankle_x, acc_ankle_y, acc_ankle_z,
	       gyro_ankle_x, gyro_ankle_y, gyro_ankle_z, mag_ankle_x, mag_ankle_y, mag_ankle_z,
	        acc_arm_x, acc_arm_y, acc_arm_z, gyro_arm_x, gyro_arm_y, gyro_arm_z, mag_arm_x, mag_arm_y, mag_arm_z)
	st.success('The output is {}'.format(result))
	if result == 0:
		st.markdown("### Activity:🚶‍♀️ Walking!")
	if result == 1:
		st.markdown("### Activity: 🏃‍♀️ Running!")