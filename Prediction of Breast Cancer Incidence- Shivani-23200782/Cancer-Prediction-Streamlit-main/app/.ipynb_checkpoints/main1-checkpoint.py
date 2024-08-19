import streamlit as st 
import pickle
import numpy as np  
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

# Set the page configuration
st.set_page_config(
    page_title="Prediction of Breast Cancer Incidence",
    page_icon=":female-doctor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to clean and preprocess data
def get_clean_data():
    data = pd.read_csv("data/data.csv")
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    return data

# Function to add the sidebar for user input
def add_sidebar():
    st.sidebar.header("Change the nuclei measurements here:")
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
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    
    return input_dict

def add_prediction(input_data):
    model = pickle.load(open("model/model.pkl","rb"))
    scaler = pickle.load(open("model/scaler.pkl","rb"))

    input_np = np.array(list(input_data.values())).reshape(1,-1)
    input_scaled = scaler.transform(input_np)

    prediction = model.predict(input_scaled)
    st.subheader("Cell Cluster Prediction")
    st.write("The cell cluster prediction is:")

    if prediction[0] == 0:
        st.write("<span class='diagnosis benign'>Benign</span>",unsafe_allow_html=True)
    else:
        st.write("<span class='diagnosis malignant'>Malignant</span>",unsafe_allow_html=True)
    
    st.write("Benign Probability: ", round(model.predict_proba(input_scaled)[0][0],3))
    st.write("Malignant Probability: ", round(model.predict_proba(input_scaled)[0][1],3))

    st.write('This analysis is based on the model trained, and is meant to provide insights on the model\'s capability in detecting whether the cell nuclei is cancerous or not.')

# Function to scale the input values
def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['diagnosis'], axis=1)
    scaled_dict = {}

    for key, value in input_dict.items():
        max_value = X[key].max()
        min_value = X[key].min()
        scaled_value = (value - min_value) / (max_value - min_value)
        scaled_dict[key] = scaled_value
    return scaled_dict

# Function to generate a radar chart
def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ["Radius", "Texture", "Perimeter", "Area",
                  "Smoothness", "Compactness", "Concavity", "Concave points",
                  "Symmetry", "Fractal Dimension"]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_mean'], input_data['texture_mean'], input_data['perimeter_mean'],
            input_data['area_mean'], input_data['smoothness_mean'], input_data['compactness_mean'],
            input_data['concavity_mean'], input_data['concave points_mean'], input_data['symmetry_mean'],
            input_data['fractal_dimension_mean']
        ],
        theta=categories,
        fill='toself',
        name='Mean'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_se'], input_data['texture_se'], input_data['perimeter_se'], input_data['area_se'],
            input_data['smoothness_se'], input_data['compactness_se'], input_data['concavity_se'],
            input_data['concave points_se'], input_data['symmetry_se'], input_data['fractal_dimension_se']
        ],
        theta=categories,
        fill='toself',
        name='Standard Error'
    ))
    fig.add_trace(go.Scatterpolar(
        r=[
            input_data['radius_worst'], input_data['texture_worst'], input_data['perimeter_worst'],
            input_data['area_worst'], input_data['smoothness_worst'], input_data['compactness_worst'],
            input_data['concavity_worst'], input_data['concave points_worst'], input_data['symmetry_worst'],
            input_data['fractal_dimension_worst']
        ],
        theta=categories,
        fill='toself',
        name='Worst'
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )),
        showlegend=True
    )

    return fig

# Function to preprocess and predict the uploaded image
def predict_uploaded_image(uploaded_file, model):
    image_size = (224, 224)  # Adjust this size to what your model expects
    
    # Load the image
    image = load_img(uploaded_file, target_size=image_size)
    image_array = img_to_array(image) / 255.0  # Normalize the image
    image_array = np.expand_dims(image_array, axis=0)  # Reshape to match model input shape

    # Predict the class
    prediction = model.predict(image_array)
    
    threshold = 0.5
    predicted_class = 1 if prediction[0][1] >= threshold else 0

    # Return class name based on the predicted class
    return "Malignant" if predicted_class == 1 else "Benign"

# New homepage with two options: Image Analysis or Cell Nuclei Analysis
def homepage():
    st.markdown("""
        <style>
        .centered-title {
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 0;
        }
        .centered-info {
            text-align: center;
            font-size: 16px;
            margin-top: 0;
        }
        .button-container {
            display: flex;
            justify-content: space-between;
            margin-top: 50px;
        }
        .button-left, .button-right {
            width: 45%;
        }
        </style>
        """, unsafe_allow_html=True)

    # Title centered
    st.markdown('<p class="centered-title">Prediction of Breast Cancer Incidence</p>', unsafe_allow_html=True)
    
    # Author info centered
    st.markdown('<p class="centered-info">Created by : Shivani, 23200782<br>MSc Data and Computational Science, UCD, Dublin, Ireland</p>', unsafe_allow_html=True)
    
    # Display buttons for navigation
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Image Analysis"):
            st.session_state.image_analysis = True
            st.session_state.show_input_form = False
    with col2:
        if st.button("Cell Nuclei Analysis"):
            st.session_state.image_analysis = False
            st.session_state.show_input_form = True

# Main function to run the app
def main():
        # Ensure the session state is initialized
    if 'show_input_form' not in st.session_state:
        st.session_state.show_input_form = False

    if 'image_analysis' not in st.session_state:
        st.session_state.image_analysis = False

    # Check if the user selected Cell Nuclei Analysis
    if st.session_state.show_input_form:
        # Cell Nuclei Analysis page
        input_data = add_sidebar()
        with st.container():
            st.title("Prediction of Breast Cancer Incidence using Cell Nuclei Measurements")

        col1, col2 = st.columns([4, 1])
        with col1:
            radar_chart = get_radar_chart(input_data)
            st.plotly_chart(radar_chart)
        with col2:
            add_prediction(input_data)

        # Add a "Home" button at the bottom
        if st.button("Home"):
            st.session_state.show_input_form = False  # Reset the state to go back to the homepage

    # Check if the user selected Image Analysis
    elif st.session_state.image_analysis:
        # Image Analysis page
        st.title("Please upload the image for analysis")
        st.write("Click the button below to upload an image for analysis.")
        
        uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "png", "jpeg"])
        
        if uploaded_file is not None:
            # Load the model only when the image is uploaded
            model = load_model("best_model1.keras")
            st.image(uploaded_file, caption="Uploaded Image.", use_column_width=True)
            prediction = predict_uploaded_image(uploaded_file, model)
            st.write(f"**Prediction:** The uploaded image is classified as **{prediction}**.")

            # Option to re-upload or go back to home
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Re-upload"):
                    st.experimental_rerun()  # Rerun the app to clear the file_uploader
            with col2:
                if st.button("Home"):
                    st.session_state.image_analysis = False  # Go back to the homepage

    # Default to showing the homepage if no other state is triggered
    else:
        homepage()

if __name__ == '__main__':
    main()

