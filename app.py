import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
import numpy as np
import tempfile

# Import your recommendation text from recommendation.py
from recommendation import cnv, dme, drusen, normal

# -----------------------
# Function: Model Prediction
# -----------------------
def model_prediction(test_image_path):
    try:
        # Load model
        model = tf.keras.models.load_model(r"C:\xray disease\Human_Eye_Disease_Prediction\model\Trained_Model.keras")

        # Load and preprocess image
        img = tf.keras.utils.load_img(test_image_path, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Predict
        predictions = model.predict(x)
        return np.argmax(predictions)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return -1

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Identification"])

# -----------------------
# Home Page
# -----------------------
if app_mode == "Home":
    st.markdown("""
    ## **OCT Retinal Analysis Platform**

#### **Welcome to the Retinal OCT Analysis Platform**
**Optical Coherence Tomography (OCT)** is a powerful imaging technique that provides high-resolution cross-sectional images of the retina, allowing for early detection and monitoring of various retinal diseases.

##### **Key Features**
- **Automated Image Analysis**: Classifies OCT images into **Normal**, **CNV**, **DME**, and **Drusen**.
- **Streamlined Workflow**: Upload, analyze, and review OCT scans easily.

#### **Retinal Diseases**
1. **Choroidal Neovascularization (CNV)**: Subretinal fluid and neovascular membrane.
2. **Diabetic Macular Edema (DME)**: Retinal thickening and intraretinal fluid.
3. **Drusen (Early AMD)**: Presence of drusen deposits.
4. **Normal Retina**: Preserved foveal contour without fluid.

#### **Get Started**
- Upload OCT Images for analysis.
- Explore results with detailed diagnostic insights.
""")

# -----------------------
# About Page
# -----------------------
elif app_mode == "About":
    st.header("About the Project")
    st.markdown("""
Retinal OCT (Optical Coherence Tomography) captures high-resolution cross sections of the retina. Our dataset contains **84,495 images** categorized into **CNV, DME, Drusen, and Normal**, verified through multiple expert grading layers to ensure accuracy.

**Dataset Sources**: Shiley Eye Institute, California Retinal Research Foundation, Medical Center Ophthalmology Associates, Shanghai First People’s Hospital, Beijing Tongren Eye Center.

Each image passed a tiered verification process by graders, ophthalmologists, and senior retinal specialists to ensure accurate labeling.
""")

# -----------------------
# Disease Identification Page
# -----------------------
elif app_mode == "Disease Identification":
    st.header("Retinal OCT Disease Identification")

    # Upload image
    test_image = st.file_uploader("Upload your OCT Image", type=["jpg", "jpeg", "png"])
    
    if test_image is not None:
        # Save to a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
            tmp_file.write(test_image.read())
            temp_file_path = tmp_file.name
        
        st.image(test_image, caption="Uploaded Image", use_container_width=True)


        # Predict button
        if st.button("Predict"):
            with st.spinner("Analyzing image..."):
                result_index = model_prediction(temp_file_path)
                
                if result_index == -1:
                    st.error("Prediction failed. Please check the image or model.")
                else:
                    class_name = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
                    st.success(f"Prediction: **{class_name[result_index]}**")

                    # Recommendation / Details
                    with st.expander("Learn More"):
                        if result_index == 0:
                            st.write("OCT scan showing *CNV with subretinal fluid.*")
                            st.markdown(cnv)
                        elif result_index == 1:
                            st.write("OCT scan showing *DME with retinal thickening and intraretinal fluid.*")
                            st.markdown(dme)
                        elif result_index == 2:
                            st.write("OCT scan showing *Drusen deposits in early AMD.*")
                            st.markdown(drusen)
                        elif result_index == 3:
                            st.write("OCT scan showing a *normal retina with preserved foveal contour.*")
                            st.markdown(normal)
