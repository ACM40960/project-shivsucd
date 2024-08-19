
import streamlit as st
import pandas as pd
import numpy as np  
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({ 'M': 1, 'B': 0 })
    return data

def add_sidebar():
    st.sidebar.header("Cell Nuclei Measurements")
    data = get_clean_data()
    
    slider_labels = [
        ("Radius (mean)", "radius_mean"),
        ("Texture (mean)", "texture_mean"),
        ("Perimeter (mean)", "perimeter_mean"),
        ("Area (mean)", "area_mean"),
        ("Smoothness (mean)", "smoothness_mean"),
        ("Compactness (mean)", "compactness_mean"),
        ("Concavity (mean)", "concavity_mean"),
        ("Concave points (mean)", "concave points_mean"),
        ("Symmetry (mean)", "symmetry_mean"),
        ("Fractal dimension (mean)", "fractal_dimension_mean"),
        ("Radius (se)", "radius_se"),
        ("Texture (se)", "texture_se"),
        ("Perimeter (se)", "perimeter_se"),
        ("Area (se)", "area_se"),
        ("Smoothness (se)", "smoothness_se"),
        ("Compactness (se)", "compactness_se"),
        ("Concavity (se)", "concavity_se"),
        ("Concave points (se)", "concave points_se"),
        ("Symmetry (se)", "symmetry_se"),
        ("Fractal dimension (se)", "fractal_dimension_se"),
        ("Radius (worst)", "radius_worst"),
        ("Texture (worst)", "texture_worst"),
        ("Perimeter (worst)", "perimeter_worst"),
        ("Area (worst)", "area_worst"),
        ("Smoothness (worst)", "smoothness_worst"),
        ("Compactness (worst)", "compactness_worst"),
        ("Concavity (worst)", "concavity_worst"),
        ("Concave points (worst)", "concave points_worst"),
        ("Symmetry (worst)", "symmetry_worst"),
        ("Fractal dimension (worst)", "fractal_dimension_worst"),
    ]

    input_dict = {}

    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(label, float(data[key].min()), float(data[key].max()), float(data[key].mean()))

    return input_dict

def load_image(image_file):
    img = Image.open(image_file)
    return img

def predict_image(model, image):
    img = image.resize((224, 224))  # Assuming the model expects 224x224 input size
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)
    return "Malignant" if prediction[0][0] > 0.5 else "Benign"

def main():
    st.title("Breast Cancer Detection and Classification")

    # Image upload for classification
    st.header("Upload a Mammogram Image")
    image_file = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"])

    if image_file is not None:
        img = load_image(image_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        
        model = load_model("model/breast_cancer_model.keras")
        prediction = predict_image(model, img)
        st.success(f"The uploaded image is classified as: {prediction}")

    # Patient data input for prediction
    st.header("Enter Patient Data")
    age = st.number_input("Age", min_value=0, max_value=120, value=30)
    smoking = st.selectbox("Smoking Status", ["Non-Smoker", "Smoker", "Former Smoker"])
    alcohol = st.selectbox("Alcohol Consumption", ["Non-Drinker", "Drinker"])
    exercise = st.selectbox("Exercise Regularly", ["Yes", "No"])

    # Placeholder for the actual prediction model logic based on the patient data
    st.write("Based on the input data, a prediction model will determine the risk of cancer.")
    st.warning("This part of the prediction is not implemented yet.")

    # Sidebar with cell nuclei measurements
    st.sidebar.header("Cell Nuclei Measurements for Custom Input")
    input_features = add_sidebar()

    # Placeholder for the actual prediction model logic based on the nuclei measurements
    st.write("Based on the input measurements, a prediction model will determine the classification.")
    st.warning("This part of the prediction is not implemented yet.")

if __name__ == '__main__':
    main()
