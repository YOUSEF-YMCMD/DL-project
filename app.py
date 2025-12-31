import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🩺",
    layout="centered"
)

# ---------------- Title ----------------
st.title("🩺 Skin Cancer Detection System")
st.write("CNN-based classification of skin lesions (Benign vs Malignant)")

# ---------------- Load Model ----------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model.h5")

model = load_model()

# ---------------- Upload Image ----------------
uploaded_file = st.file_uploader(
    "📤 Upload a skin lesion image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- Image Preprocessing ----------------
def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ---------------- Main Logic ----------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    # -------- Input Validation (Resolution) --------
    if image.size[0] < 100 or image.size[1] < 100:
        st.warning("⚠ Image resolution is too low. Please upload a clearer skin image.")
        st.stop()

    col1, col2 = st.columns([1, 1])

    # -------- Column 1: Image --------
    with col1:
        st.subheader("🖼 Uploaded Image")
        st.image(image, caption="Skin Lesion", width=280)

    # -------- Column 2: Analysis --------
    with col2:
        st.subheader("📊 Analysis Result")

        if st.button("🔍 Analyze Image"):
            with st.spinner("Analyzing image..."):
                img_array = preprocess_image(image)
                prob = model.predict(img_array)[0][0]
                percentage = prob * 100

            # -------- Non-Skin Image Rejection --------
            # -------- Improved Non-Skin Check --------
            if percentage > 99.2:
                st.warning(
                    "⚠ The uploaded image may not be a valid skin lesion image. "
                    "Please ensure the image is clear and focused on skin."
                )
                st.stop()

            # -------- Result Visualization --------
            st.metric(
                label="Probability of Malignancy",
                value=f"{percentage:.1f}%"
            )
            st.progress(int(percentage))

            # -------- Decision & Explanation --------
            if prob > 0.5:
                st.error("🔴 **Malignant**")
                st.warning(
                    "High-risk lesion detected. "
                    "Further medical examination is strongly recommended."
                )
            else:
                st.success("🟢 **Benign**")
                st.info(
                    "Low-risk lesion detected. "
                    "Regular monitoring is advised."
                )

# ---------------- Footer ----------------
st.markdown("---")
st.caption(
    "Developed by Yousef Mohamed | CNN • TensorFlow • Streamlit  \n"
    "Educational Use Only — Not a Medical Diagnosis Tool"
)
