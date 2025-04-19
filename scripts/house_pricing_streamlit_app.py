import streamlit as st

#st.title("Housing Prices Prediction")

st.image("[image_file_location]/image_Iowa.png", caption="Iowa, USA", use_container_width=True) #replace [image_file_location] with actual location of image

st.write("""
### Would you want to know?

According to Wikipedia, Iowa, Ames is *"known for its robust, stable economy, flourishing cultural environment, comprehensive medical care, top-quality schools, acres of parks and recreational opportunities, and the world-renowned Iowa State University."*  

A thriving town, with a significant attraction to its university (students make half of its population), home ownership in Ames could be a wise investment.  If **between 2006 and 2010** you purchased or sold a home in this college town, you might be curious to know if you got a fair price.  Or maybe you thought about purchasing but didn't.  Was it a missed opportunity? <span style="color: #B71C1C">**LET'S FIND OUT!**</span>

In this application you can enter some details about the home and receive a historical estimation of the price.  The estimation is based on almost 3,000 actual residential housing sale transactions from Ames, Iowa from 2006 to 2010, compiled from the Ames Assessor’s Office by Dean De Cock.  

""",unsafe_allow_html=True
)


import pickle
model = pickle.load(open('[your_file_location]/trained_pipe_randomforestregressor.sav', 'rb')) #replace [your_file_location] with actual file location

##Use the following if do not want to use the slider (value sliding option) and want to enter the actual figure:
#LotArea = st.number_input("How large was the lot (in ft²)?")
#TotalBsmtSF = st.number_input("How large was the basement (in ft²)?")
#BedroomAbvGr = st.number_input("How many bedrooms were there (not including the basement)?")
#GarageCars = st.number_input("How many cars could fit in the garage?")


LotArea = st.slider("Select the lot size (in ft²):", min_value=1300, max_value=215245, value=(215245), step=50)

TotalBsmtSF = st.slider("Select the basement size (in ft²):", min_value=0, max_value=6110, value=(50000), step=50)

BedroomAbvGr = st.slider("Select the number of bedrooms (not including the basement):", min_value=0, max_value=8, value=(8), step=1)

GarageCars = st.slider("Select the number of cars that could fit in the garage:", min_value=0, max_value=4, value=(4), step=1)


import pandas as pd
import numpy as np
new_house = pd.DataFrame({
    'LotArea':[LotArea],
    'TotalBsmtSF':[TotalBsmtSF],
    'BedroomAbvGr':[BedroomAbvGr],
    'GarageCars':[GarageCars]
})

prediction = model.predict(new_house)

# #if don't want to round up, could replace the below two lines with this one:
# st.write("Based on our data, we estimate the home would have been worth:", prediction)

# # Rounding up the prediction before displaying it
prediction_rounded = np.ceil(prediction[0])  # Assuming prediction returns an array-like object

# #Display the rounded result
# st.write("Based on our data, we estimate the home would have been worth: $", prediction_rounded)


# Formatting the prediction to have two decimal places
prediction_formatted = f"{prediction_rounded:,.2f}"  # 2 decimal place and comma for thousands

# Displaying the result with larger font size and color green using HTML, and formatted result
st.write(f"Based on our data, we estimate the home would have been worth: <span style='font-size: 30px; color: green;'>${prediction_formatted}</span>", unsafe_allow_html=True)

st.write("""
### So...did you get a fair price?
""")

st.markdown("""
------------
### **Model used for this estimation -**
For the estimations in this application I applied a RobustTreeRegressor model for supervised machine learning in Python with an r2 score of 0.73.
""")
