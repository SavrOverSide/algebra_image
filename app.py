import streamlit as st
import numpy as np
import cv2

# Function to compress image using SVD
def compress_image_with_svd(image, num_singular_values):
    # Convert image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply SVD to the grayscale image
    U, S, Vt = np.linalg.svd(gray_image, full_matrices=False)
    
    # Retain only the specified number of singular values
    S[num_singular_values:] = 0
    
    # Reconstruct the compressed image
    compressed_image = np.dot(U, np.dot(np.diag(S), Vt))
    
    # Normalize the compressed image data
    compressed_image = (compressed_image - np.min(compressed_image)) / (np.max(compressed_image) - np.min(compressed_image))
    
    return compressed_image

# Streamlit app
st.title('Image Compression using SVD')

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    image = cv2.imdecode(np.fromstring(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Select number of singular values for compression
    num_singular_values = st.slider('Select the number of singular values to retain:', 1, min(image.shape), 50)
    
    # Compress the image using SVD
    compressed_image = compress_image_with_svd(image, num_singular_values)
    
    # Display the compressed image
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)