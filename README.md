# Handwritten Digit Recognition

A CNN-based handwritten digit recognition system built using Python, TensorFlow/Keras, and the MNIST dataset. The application provides a Streamlit web interface where users can upload a handwritten digit image and receive the predicted digit with a confidence score.

## Features

* Handwritten digit recognition from 0 to 9
* MNIST dataset for training and testing
* Image preprocessing and pixel normalization
* Convolutional Neural Network (CNN)
* Model evaluation with accuracy and loss
* Upload handwritten digit images through Streamlit
* Prediction confidence score
* MNIST prediction visualization

## Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pillow
* Matplotlib
* Streamlit
* MNIST Dataset
* Convolutional Neural Network (CNN)

## Project Structure

```text
Handwritten Digit Recognition/
│
├── app.py
├── train_model.py
├── predict.py
├── requirements.txt
├── README.md
├── .gitignore
├── mnist_predictions.png
│
├── models/
│   └── digit_model.keras
│
└── venv/
```

> The `venv` folder is used only for the local Python environment and is excluded from GitHub using `.gitignore`.

## How It Works

```text
MNIST Dataset
      ↓
Data Preprocessing
      ↓
Pixel Normalization
      ↓
CNN Model
      ↓
Model Training
      ↓
Model Evaluation
      ↓
Upload Handwritten Digit
      ↓
Image Preprocessing
      ↓
Digit Prediction
      ↓
Confidence Score
```

## Model Architecture

The project uses a Convolutional Neural Network containing:

* Input layer: 28 × 28 × 1
* Convolutional layer: 32 filters
* Max Pooling layer
* Convolutional layer: 64 filters
* Max Pooling layer
* Flatten layer
* Dense layer: 128 neurons
* Dropout layer
* Output layer: 10 classes (0–9)

## Model Performance

The trained CNN achieved approximately:

**Test Accuracy: 99.01%**

The model is saved as:

```text
models/digit_model.keras
```

## Installation

Clone the repository:

```bash
git clone https://github.com/anupriyaerigala22-del/Handwritten-Digit-Recognition.git
```

Go to the project directory:

```bash
cd Handwritten-Digit-Recognition
```

Create a virtual environment:

```bash
python -m venv venv
```

Activate the virtual environment on Windows:

```powershell
venv\Scripts\activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Train the Model

If you want to train the model again, run:

```bash
python train_model.py
```

This downloads the MNIST dataset, trains the CNN model, evaluates it, and saves the trained model in the `models` folder.

## Run the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application allows you to upload an image containing a handwritten digit and displays the predicted digit and model confidence.

## Example

Input:

```text
Handwritten digit image → 7
```

Output:

```text
Predicted Digit: 7
Model Confidence: 99.97%
```

## Learning Outcomes

Through this project, the following concepts were implemented:

* Understanding the MNIST dataset
* Image preprocessing
* Pixel normalization
* Convolutional Neural Networks
* Neural network training
* Model evaluation
* Image classification
* Model prediction
* Streamlit application development

## Limitations

This project is trained specifically on the MNIST handwritten digit dataset. Therefore, it recognizes **single handwritten digits from 0 to 9** and does not recognize letters or multiple characters in one image.

## Future Enhancements

* Add handwritten character recognition
* Support multiple digit detection
* Improve image preprocessing
* Add real-time drawing canvas
* Deploy the application online
* Add prediction history

## Author

**Anupriya Erigala**

## License

This project is created for educational and internship purposes.
