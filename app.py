import streamlit as st
import numpy as np
import cv2
import tensorflow as tf

st.set_page_config(
    page_title="Malaria Detection System",
    layout="wide",
    initial_sidebar_state="expanded"
)

model = tf.keras.models.load_model("malaria_model.h5")
IMG_SIZE = 96

# ---------------- Sidebar ----------------
st.sidebar.title("Model Information")
st.sidebar.markdown("""
**Model Type:** CNN  
**Validation Accuracy:** 93%  
**Classes:** Parasitized / Uninfected  
**Dataset:** NIH Malaria Cell Images  
""")

# ---------------- Header ----------------
st.title("Malaria Detection System")
st.caption("Deep Learning Based Microscopic Blood Cell Classification")

st.markdown("---")

# ---------------- Upload Section ----------------
uploaded_file = st.file_uploader(
    "Upload Microscopic Blood Cell Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    col1, col2 = st.columns([1, 1.2])

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    # Display Image
    with col1:
        st.subheader("Uploaded Image")
        st.image(img, width=320)

    # Prediction
    with st.spinner("Analyzing image..."):
        img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img_resized = img_resized / 255.0
        img_resized = np.reshape(img_resized, (1, IMG_SIZE, IMG_SIZE, 3))
        prediction = model.predict(img_resized)[0][0]

    prob_uninfected = float(prediction)
    prob_parasitized = float(1 - prediction)

    # Display Result
    with col2:
        st.subheader("Prediction Result")

        if prediction > 0.5:
            st.success("Uninfected")
        else:
            st.warning("Parasitized")

        st.markdown("#### Confidence Scores")

        st.progress(prob_parasitized)
        st.write(f"Parasitized: {prob_parasitized*100:.2f}%")

        st.progress(prob_uninfected)
        st.write(f"Uninfected: {prob_uninfected*100:.2f}%")

st.markdown("---")
st.caption("Developed using TensorFlow and Streamlit")