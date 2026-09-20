import sys
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


MODEL_PATH = "models/digit_model.keras"

# Load trained model
model = tf.keras.models.load_model(MODEL_PATH)


def prepare_image(image_path):
    """
    Prepare the uploaded image for the MNIST CNN model.
    """

    # Open image and convert to grayscale
    image = Image.open(image_path).convert("L")

    # Resize to MNIST size
    image = image.resize((28, 28))

    # Convert image to numpy array
    image_array = np.array(image)

    # Invert colors
    # MNIST: black background + white digit
    image = ImageOps.invert(Image.fromarray(image_array))
    image_array = np.array(image)

    # Normalize pixel values
    image_array = image_array.astype("float32") / 255.0

    # Add batch and channel dimensions
    image_array = image_array.reshape(1, 28, 28, 1)

    return image_array


def predict_digit(image_path):
    """
    Predict the handwritten digit and return
    the digit and confidence.
    """

    image = prepare_image(image_path)

    predictions = model.predict(image, verbose=0)

    predicted_digit = int(np.argmax(predictions))

    confidence = float(np.max(predictions)) * 100

    return predicted_digit, confidence


# Allow prediction from terminal
if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("Please provide an image path.")
        print("Example:")
        print("python predict.py digit.png")

        sys.exit()

    image_path = sys.argv[1]

    digit, confidence = predict_digit(image_path)

    print(f"Predicted Digit: {digit}")
    print(f"Confidence: {confidence:.2f}%")