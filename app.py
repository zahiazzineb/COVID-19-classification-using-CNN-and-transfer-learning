import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('modelcovid.h5')

# Define the categories
categories = ['COVID-19', 'Non-COVID-19']

# Create a Streamlit app
st.set_page_config(page_title="Chelbi Group LABs \n COVID-19 Classification Model", 
                   page_icon="", 
                   layout="wide")
# Add a logo
logo = Image.open('Fichier 1-8.png')  # replace with your logo file
st.image(logo, width=100)

# Add a header with your name
st.header("Chelbi Group LABs")

# Create a file uploader
uploaded_file = st.file_uploader("Upload a chest TDM image:", type=["jpg", "jpeg", "png"])

# Create a button to make a prediction
if uploaded_file is not None:
    st.write("Uploaded image:")
    st.image(uploaded_file, width=300)

    # Preprocess the image
    img = Image.open(uploaded_file)
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Make a prediction
    prediction = model.predict(img)
    score = tf.nn.softmax(prediction)

    # Display the result
    st.write("Prediction:")
    st.write(f"COVID-19: {score[0][0]*100:.2f}%")
    st.write(f"Non-COVID-19: {score[0][1]*100:.2f}%")
    st.write(f"Classified as: {categories[np.argmax(score)]}")