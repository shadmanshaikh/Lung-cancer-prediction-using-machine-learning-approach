# streamlit is a python library used for displaying manipulated data , working with different kinds of charts 
# sklearn aka sci-kit learn used for building various kind of ANN pipelines , used for variety of tasks like preprocessing of data , consists of classification models 
# pandas is majorly used for data I/O , reading , writing and manipulation of data .
# numpy is used for data manipulation and convertion of vectors
# PIL is a py library used for I/O images to display

from random import random
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image

# heading of the topic
st.write("# :computer: Machine Learning Model of Lung Cancer Prediction")

# importing of the dummy image used for the title 
image = Image.open('bc_diagnosis.jpg')
st.image(image, caption='Doctor Diagnosing a patient')

# reading the cancer survey dataf
df = pd.read_csv('survey.csv')

x = df.iloc[:,:-1].values
y = df.iloc[:,-1].values

# taking first five rows to show the data after the encoding
st.subheader("sample data that is being used")
# inp = df.iloc[:5,:-1].values
first5 = df.head(5)
st.write(first5)

le =  LabelEncoder()
# st.write(x)
y = le.fit_transform(y) #yes 1 no 0
x[:,0] = le.fit_transform(x[:,0]) #male 1 female 0 
# st.write(y)
# st.write(x)

#sample data that is being used



#general stats related to the dataset
st.subheader('# Stats :')
st.write(df.describe())

# st.subheader("graphical representation: ")
# st.altair_chart(df)

# splitting the data for testing and training
x_train, x_test, y_train , y_test = train_test_split(x , y, test_size=0.20 , random_state=0)

# getting the user inputs on all the attributes listed in the dataset using slidebars 
def get_user_input():
    gender = st.sidebar.slider('gender: male(1) & female(0)' , 0 , 1 , 0)
    age = st.sidebar.slider('age', 21 , 87,30)
    smoking = st.sidebar.slider('smoking',1 , 2 ,1)
    yellow_fingers = st.sidebar.slider('yellow_fingers' , 1 , 2 , 1)
    anxiety = st.sidebar.slider('anxiety' , 1 , 2 , 1)
    peer_pressure = st.sidebar.slider('peer_pressure' , 1 , 2 , 1)
    chronic_desease = st.sidebar.slider('chronic_desease' , 1 , 2 , 1)
    fatigue = st.sidebar.slider('fatigue'  , 1 , 2 , 1)
    allergy = st.sidebar.slider('allergy', 1 , 2 , 1)
    wheezing = st.sidebar.slider('wheezing', 1 , 2 , 1)
    alcohol_consuming = st.sidebar.slider('alcohol_consuming', 1 , 2 , 1)
    coughing = st.sidebar.slider('coughing', 1 , 2 , 1)
    shortness_of_breath = st.sidebar.slider('sob', 1 , 2 , 1)
    swallowing_difficulty = st.sidebar.slider('sd', 1 , 2 , 1)
    chest_pain = st.sidebar.slider('chest_pain', 1 , 2 , 1)

    user_data = {'gender':gender,
                 'age':age,
                 'smoking': smoking,
                 'yellow_fingers': yellow_fingers,
                 'anxiety': anxiety,
                 'peer_pressure':peer_pressure,
                 'chronic_desease':chronic_desease,
                 'fatigue':fatigue,  
                 'allergy':allergy,  
                 'alcohol_consuming':alcohol_consuming, 
                 'coughing':coughing, 
                 'wheezing':wheezing, 
                 'shortness_of_breath':shortness_of_breath,  
                 'swallowing_difficulty':swallowing_difficulty,  
                 'chest_pain':chest_pain,  
                    }
    
    features = pd.DataFrame(user_data,index=[0])

    return features

# show the user input in realtime
user_input = get_user_input()
st.subheader("showing the user inputs")
st.write(user_input)

# applying random forest classifier for training and testing the data
RandomForestClassifier =RandomForestClassifier()
RandomForestClassifier.fit(x_train,y_train)


st.subheader("the result is :")
predict = RandomForestClassifier.predict(user_input)

if predict == 0:
    st.write("## You seem healthy 👍")
else:
    st.write("## Seems like you need to consult a doctor 😢")


#accuracy score

st.subheader('the accuracy score is: ')
st.write(str(accuracy_score(y_test,RandomForestClassifier.predict(x_test))*100)+'%')


st.write("""
## A Project by [shadman](www.github.com/shady4real)
""")
