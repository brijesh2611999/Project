import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load the model
model = tf.keras.models.load_model('mnist_model.h5')

# Define a function to preprocess the image
def preprocess_image(image):
    img = Image.open(image).convert('L')  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to MNIST input size
    img = np.array(img) / 255.0  # Normalize pixel values
    img = img.reshape(1, 28, 28, 1)  # Reshape for model input
    return img

# Define the Streamlit app
def main():
    st.title('MNIST Digit Recognition')
    st.write('Upload a handwritten digit image to classify it.')

    # Upload image
    uploaded_image = st.file_uploader('Upload Image', type=['jpg', 'png', 'jpeg'])

    if uploaded_image is not None:
        st.image(uploaded_image, caption='Uploaded Image', use_column_width=True)
        prediction_button = st.button('Predict')

        if prediction_button:
            img_array = preprocess_image(uploaded_image)
            prediction = model.predict(img_array)
            predicted_digit = np.argmax(prediction)
            st.write(f'Predicted Digit: {predicted_digit}')

if __name__ == '__main__':
    main()







































# from flask import Flask, request, jsonify, render_template
# from tensorflow.keras.models import load_model
# import numpy as np
# from PIL import Image
# import io
#
# app = Flask(__name__)
#
# # Load the pre-trained model
# model = load_model('mnist_my_model.keras')
#
#
# @app.route('/')
# def index():
#     return render_template('index.html')
#
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Receive uploaded image
#     file = request.files['image']
#
#     # Log that the image is received
#
#     app.logger.info('Image received: %s', file.filename)
#
#     # Read the image file and preprocess
#     img = Image.open(io.BytesIO(file.read())).convert('L')  # Convert to grayscale
#     img = img.resize((28, 28))  # Resize to MNIST input size
#     img = np.array(img) / 255.0  # Normalize pixel values
#     img = img.reshape(1, 28, 28, 1)  # Reshape for model input
#
#     # Make prediction
#     prediction = model.predict(img)
#     predicted_digit = np.argmax(prediction)
#
#     # Convert predicted_digit to regular Python integer
#     predicted_digit = int(predicted_digit)
#
#     # Return prediction to client
#     return jsonify({'prediction': predicted_digit})
#
#
# if __name__ == '__main__':
#     app.run(debug=True)
