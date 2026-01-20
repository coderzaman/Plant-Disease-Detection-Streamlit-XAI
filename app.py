import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- 1. App Configuration ---
st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")

# --- 2. Load the Model (Cached for Speed) ---
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('best_custom_cnn.keras')
    return model

model = load_model()

# --- 3. Class Names (38 Classes from PlantVillage) ---
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    'Tomato___healthy'
]

# --- 4. Image Preprocessing Function ---
def preprocess_image(image):
    # Resize to 224x224 (Model Input Shape)
    image = image.resize((224, 224))
    # Convert to Array
    img_array = np.array(image)
    # Normalize (0-1) - Same as Custom CNN Training
    img_array = img_array / 255.0
    # Add Batch Dimension (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- 5. UI Layout ---
st.title("🌿 Plant Disease Detection System")
st.markdown("Upload an image of a plant leaf to detect diseases.")

# File Uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the Image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf Image', use_column_width=True)
    
    # Predict Button
    if st.button('Analyze Image'):
        with st.spinner('Analyzing... Please wait...'):
            try:
                # Preprocess
                processed_image = preprocess_image(image)
                
                # Prediction
                predictions = model.predict(processed_image)
                predicted_class = class_names[np.argmax(predictions)]
                confidence = np.max(predictions) * 100
                
                # Display Result
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"**Confidence:** {confidence:.2f}%")
                
                # Basic Health Status Logic
                if "healthy" in predicted_class:
                    st.balloons()
                    st.success("✅ The plant is healthy!")
                else:
                    st.warning("⚠️ Disease Detected! Consult an agriculturist.")
                    
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
st.caption("Developed for Thesis Research | Using Custom Lightweight CNN")
