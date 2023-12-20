#kaler biru ni dlm linux, base python and writing utk generate py file
import streamlit as st
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import GaussianNB

st.write("# Simple Iris Flower Prediction App") # kena ada # utk first title
st.write("This app predicts the **Iris flower** type!")

st.sidebar.header('User Input Parameters') #sidebar utk user interact

def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length', 4.3, 7.9, 5.4) # value max , min, default
    sepal_width = st.sidebar.slider('Sepal width', 2.0, 4.4, 3.4) #user define function utk user interaction
    petal_length = st.sidebar.slider('Petal length', 1.0, 6.9, 1.3)
    petal_width = st.sidebar.slider('Petal width', 0.1, 2.5, 0.2)
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0]) #data kena ikut susunan column features yang ditrain
    return features

#call function
df = user_input_features()

st.subheader('User Input parameters')
st.write(df)

#load dataset
data = sns.load_dataset('iris')
X = data.drop(['species'],axis=1)
Y = data.species.copy()

#training
modelGaussianIris = GaussianNB()
modelGaussianIris.fit(X, Y)

prediction = modelGaussianIris.predict(df)
prediction_proba = modelGaussianIris.predict_proba(df)

st.subheader('Class labels and their corresponding index number')
st.write(Y.unique())

st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
