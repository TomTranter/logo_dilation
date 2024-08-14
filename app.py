import streamlit as st
import numpy as np
from skimage import morphology
import skimage
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import io
from PIL import Image

# Streamlit app title
st.title("Image Processing with Dilation and Overlay")

# Image upload
uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Load the image
    image = skimage.io.imread(uploaded_file)
    gray_image = image[:, :, 1]

    # User inputs for processing
    threshold = st.slider("Threshold", 0, 255, 170)
    padding = st.slider("Padding (pixels)", 0, 200, 100)
    disk_size = st.slider("Structuring Element Size", 1, 20, 4)
    num_dilations = st.slider("Number of Dilation Iterations", 1, 20, 12)
    sigma = st.slider("Gaussian Blur Sigma", 0.0, 10.0, 4.0)

    # Background color picker
    background_color = st.color_picker("Pick a background color", "#ffffff")

    # Convert background color to RGB format
    bg_rgb = tuple(int(background_color[i:i+2], 16) for i in (1, 3, 5))

    # Threshold the image to create a binary image
    binary_image = gray_image < threshold

    # Pad the binary image
    padded_binary_image = np.pad(binary_image, pad_width=padding, mode="constant", constant_values=0)

    # Create a structuring element
    structuring_element = morphology.disk(disk_size)

    # Perform dilation and Gaussian blur
    dilated_image = padded_binary_image
    for i in range(num_dilations):
        dilated_image = morphology.dilation(dilated_image, structuring_element)
        dilated_image = gaussian_filter(dilated_image.astype(float), sigma=sigma)
        dilated_image = dilated_image > 0.5

    # Overlay the original image back into the center of the dilated image
    center_x = (dilated_image.shape[1] - binary_image.shape[1]) // 2
    center_y = (dilated_image.shape[0] - binary_image.shape[0]) // 2

    # Initialize an RGB image with an alpha channel for transparency
    rgb_image = np.zeros((*dilated_image.shape, 4), dtype=np.uint8)

    # Set the dilated part to white with full opacity
    rgb_image[dilated_image > 0] = [*bg_rgb, 255]  # Apply chosen background color  # White for dilated part

    # Overlay the original image (broadcast back) with full opacity
    for i in range(3):  # Apply for each color channel
        rgb_image[center_y:center_y + binary_image.shape[0], center_x:center_x + binary_image.shape[1], i][binary_image] = image[:, :, i][binary_image]
    rgb_image[center_y:center_y + binary_image.shape[0], center_x:center_x + binary_image.shape[1], 3][binary_image] = 255  # Full opacity for the original image area

    # Set the background to be fully transparent
    rgb_image[dilated_image == 0] = [0, 0, 0, 0]

    # Determine the axes limits based on the size of the images
    xlim = (0, max(image.shape[1], padded_binary_image.shape[1]))
    ylim = (0, max(image.shape[0], padded_binary_image.shape[0]))

    # Display the original and processed images
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    ax = axes.ravel()

    ax[0].imshow(image)
    ax[0].set_title('Original Image')

    ax[1].imshow(binary_image, cmap=plt.cm.gray)
    ax[1].set_title('Binary Image')

    ax[2].imshow(dilated_image, cmap=plt.cm.gray)
    ax[2].set_title('Dilated Image')

    ax[3].imshow(rgb_image)
    ax[3].set_title('Logo Image')

    for a in ax:
        a.axis('off')
    # Apply the same limits to all axes
    for a in ax:
        a.set_xlim(xlim)
        a.set_ylim(ylim[::-1])  # Reverse y-axis to match image display

    plt.tight_layout()
    
    st.pyplot(fig)
    # Save the image to an in-memory file
    # Convert the final image array to an Image object
    final_image = Image.fromarray(rgb_image)
    buf = io.BytesIO()
    final_image.save(buf, format="PNG")
    byte_im = buf.getvalue()
    # Add a download button
    st.download_button(
        label="Download Image",
        data=byte_im,
        file_name="overlay_image.png",
        mime="image/png"
    )