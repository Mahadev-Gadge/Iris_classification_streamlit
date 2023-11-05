import streamlit as st
import pandas as pd
import numpy as np
from streamlit_pandas_profiling import st_profile_report
from ydata_profiling import ProfileReport
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Project title
st.title(""" :violet[Iris dataset classification]""")
st.subheader("Project description:")   
overview=''' This is web application for classification problem implemented using streamlit library.  
                Sidebar mainly consists of two components.  
                1. Playground: Helps user to provide test datapoint for prediction.  
                2. Prameter configuration: User can select parameters and model behaves accordingly.
            '''
st.markdown(overview)        

life_cycle=''' Typically datascience projects life cycle consists of below steps. Similar approach has been carried out to solve classification problem.  
                1. Data acquisition.  
                2. Data preprocessing and visualization.  
                3. Model selection, building and evaluation.  
                4. Prediction.  
                5. Deployment.  
                
            '''
st.markdown(life_cycle)       
st.divider()
         
st.sidebar.header(""" Input dataset """)
uploaded_file=st.sidebar.file_uploader(":blue[**Upload input csv file**]")
#import pdb;pdb.set_trace()

st.sidebar.header(""" Playground """)
sepal_length = st.sidebar.slider(':violet[**Sepal_length**]', 4.3, 7.9, 5.0)
sepal_width = st.sidebar.slider(':violet[**Sepal_width**]', 2.0, 4.4, 3.0)
petal_length = st.sidebar.slider(':violet[**Petal_length**]', 0.1, 6.9, 4.0)
petal_width = st.sidebar.slider(':violet[**Petal_width**]', 0.0, 2.0, 1.0)

test_point=np.array([sepal_length, sepal_width, petal_length, petal_length]).reshape(1,-1)

#1.Input dataset
st.subheader("1. Dataset overview")
data=pd.read_csv(uploaded_file)
pr=ProfileReport(data, explorative=True)

if st.checkbox('Click here to view dataset'):
    st.write(data)
st.divider()

#2.Statiscal and visualization report
st.subheader("2. Exploratory data analysis")
if st.checkbox("Click here to view statistical and visualization report"):
    st_profile_report(pr)
st.divider()

#3.Model build
st.subheader("3. Model training")
iris = pd.read_csv(uploaded_file.name)
target_mapper = {'Setosa':0, 'Versicolor':1, 'Virginica':2}

def target_encode(val):
    return target_mapper[val]

iris['variety'] = iris['variety'].apply(target_encode)

Y = iris['variety']
X = iris.drop('variety', axis=1)

from sklearn.linear_model import LogisticRegression

st.sidebar.subheader(""" Parameter configuration """)
penalty=st.sidebar.radio(label='Penalty', key="visibility",options=["l1", "l2"])
clf = LogisticRegression(penalty= penalty, solver='liblinear')
clf.fit(X, Y)
st.write(f"**Model Name: {clf}**")
st.success("Model training is successful.", icon='✅')
st.divider()

#4.Display user provided input
user_input=pd.DataFrame(data=test_point, columns=['sepal_length','sepal_width','petal_length','petal_width'])
st.subheader("4. Test datapoint prediction")
st.write(user_input)
st.divider()

#5.Test datapoint prediction
prediction = clf.predict(test_point)
prediction_proba = clf.predict_proba(test_point)
species_prediction=pd.DataFrame(data=prediction_proba, columns=['Setosa','Versicolor','Virginica'])

st.subheader('5. Model output')
st.markdown('Prediction Probability')
st.write(species_prediction)

iris_species = np.array(['Setosa','Versicolor','Virginica'])
st.write(f'Predicted output is **{iris_species[prediction][0]}**.')

  
