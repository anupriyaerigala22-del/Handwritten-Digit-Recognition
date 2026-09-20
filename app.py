import streamlit as st
from PIL import Image
from predict import predict_digit
import os

st.set_page_config(
    page_title="Handwritten Digit Recognition",
    page_icon="🔢",
    layout="centered"
)

st.title("🔢 Handwritten Digit Recognition")

st.write(
    "Upload an image of a handwritten digit and the trained "
    "CNN model will predict the digit."
)

st.divider()

model_path = "models/digit_model.keras"

if not os.path.exists(model_path):
    st.error(
        "Trained model not found. Please run "
        "`python train_model.py` first."
    )
    st.stop()

uploaded_file = st.file_uploader(
    "Upload a handwritten digit image",
    type=["png", "jpg", "jpeg"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")

    st.image(
        image,
        caption="Input Image",
        width=250
    )

    if st.button("🔍 Predict Digit"):

        temp_path = "temp_digit.png"

        image.save(temp_path)

        try:
            digit, confidence = predict_digit(temp_path)

            st.success(f"Predicted Digit: {digit}")

            st.info(
                f"Model Confidence: {confidence:.2f}%"
            )

        except Exception as e:

            st.error(f"Prediction error: {e}")

        finally:

            if os.path.exists(temp_path):
                os.remove(temp_path)

st.divider()

st.subheader("How the System Works")

st.write(
    "Image → Preprocessing → CNN Model → Prediction → Result"
)

st.caption(
    "Technologies: Python | TensorFlow/Keras | MNIST | "
    "CNN | Streamlit"
)