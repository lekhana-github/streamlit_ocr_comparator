import streamlit as st   # Import Streamlit for web app functionality
from PIL import Image, ImageDraw   # Import Pillow for image handling and drawing on images
import numpy as np     # Import NumPy for array manipulation
import fitz  # PyMuPDF for PDF handling
import easyocr  # EasyOCR for Optical Character Recognition
from paddleocr import PaddleOCR  # PaddleOCR for Optical Character Recognition
import html    # For HTML escaping of text
import pandas as pd   # For working with DataFrames
import io    # For handling in-memory binary data


# Streamlit page title
st.markdown(
    "<h1 style='text-align: center; color: #FF8C00;'>OCR COMPARISON: EASY OCR VS PADDLE OCR</h1>",
    unsafe_allow_html=True,  # Display HTML content on the page (unsafe_allow_html=True allows custom styling)
)


# File uploader for images or PDFs
uploaded_file = st.file_uploader("Upload an Image or PDF", type=["png", "jpg", "jpeg", "pdf"])


# Sidebar for hyperparameter tuning
st.sidebar.header("Customize OCR Hyperparameters")


# Hyperparameters with explanations for fine-tuning OCR
min_size = st.sidebar.slider("Min Size", 1, 1000, 20)# Minimum text size for detection
text_threshold = st.sidebar.slider("Text Threshold", 0.0, 1.0, 0.6)# Text detection confidence threshold
link_threshold = st.sidebar.slider("Link Threshold", 0.0, 1.0, 0.4) # Link confidence for adjacent text
low_text = st.sidebar.slider("Low Text Threshold", 0.0, 1.0, 0.3)  # Threshold for low-confidence text
mag_ratio = st.sidebar.slider("Magnification Ratio", 1.0, 10.0, 2.0) # Image magnification ratio
canvas_size = st.sidebar.slider("Canvas Size", 0, 1000, 500) # Canvas size for text area
slope_ths = st.sidebar.slider("Slope Threshold", 0.0, 10.0, 0.5) # Threshold for text slope
ycenter_ths = st.sidebar.slider("Y-center Threshold", 0.0, 1.0, 0.3) # Y-axis center alignment threshold
height_ths = st.sidebar.slider("Height Threshold", 0.0, 1.0, 0.4) # Text height threshold
width_ths = st.sidebar.slider("Width Threshold", 0.0, 1.0, 0.3) # Text width threshold
add_margin = st.sidebar.slider("Add Margin", 0, 100, 10) # Margin around detected text
optimal_num_chars = st.sidebar.slider("Optimal Number of Characters", 1, 100, 10) # Optimal character length


# Function to convert PDF to a list of images
def convert_pdf_to_images(pdf_file):

    """
    Converts a PDF file into a list of images for each page.
    Args:
        pdf_file: The uploaded PDF file.

    Returns:
        A list of images for each page in the PDF.
    """

    pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
    return [
        Image.frombytes("RGB", [page.width, page.height], page.get_pixmap().samples)
        for page in pdf_document
    ]


# Function to perform OCR using EasyOCR
def perform_easyocr(image):

    """
    Performs OCR on the given image using EasyOCR with specified hyperparameters.
    Args:
        image: Input image for OCR.

    Returns:
        A list of detected text boxes and corresponding text.
    """

    reader = easyocr.Reader(["en"])
    np_image = np.array(image)

    results = reader.readtext(
        np_image,
        min_size=min_size,
        text_threshold=text_threshold,
        link_threshold=link_threshold,  # Added for better link confidence adjustment
        low_text=low_text,
        mag_ratio=mag_ratio,
        canvas_size=canvas_size,
        slope_ths=slope_ths,
        ycenter_ths=ycenter_ths,
        height_ths=height_ths,
        width_ths=width_ths,
        add_margin=add_margin / 100  # Adjusted for percentage-based margin (Streamlit slider gives integers)
    )
    return results


# Function to perform OCR using PaddleOCR
def perform_paddleocr(image):

    """
    Performs OCR on the given image using PaddleOCR with specified hyperparameters.
    Args:
        image: Input image for OCR.

    Returns:
        OCR results containing detected text and confidence levels.

    """
    
    reader = PaddleOCR(
        use_angle_cls=True, 
        min_size=min_size,
        text_threshold=text_threshold,
        link_threshold=link_threshold,  # Added for better link confidence adjustment
        low_text=low_text,
        mag_ratio=mag_ratio,
        canvas_size=canvas_size, 
        slope_ths=slope_ths, 
        ycenter_ths=ycenter_ths, 
        height_ths=height_ths, 
        width_ths=width_ths, 
        add_margin=add_margin, 
        optimal_num_chars=optimal_num_chars
    )
    np_image = np.array(image)
    results = reader.ocr(np_image, cls=True)
    return results


# Draw bounding boxes based on OCR results (Handles EasyOCR and PaddleOCR formats)
def draw_boxes(image, results, ocr_type="easyocr"):
 
    """
    Draws bounding boxes around detected text.
    Args:
        image: Input image.
        results: OCR results (EasyOCR or PaddleOCR).
        ocr_type: Type of OCR ('easyocr' or 'paddleocr').

    Returns:
        Image with bounding boxes drawn.
    """

    draw = ImageDraw.Draw(image)
    
    if ocr_type == "easyocr":
        for box, text, confidence in results:
            top_left = tuple(map(int, box[0]))
            bottom_right = tuple(map(int, box[2]))
            draw.rectangle([top_left, bottom_right], outline="red", width=3)
    else:  # For PaddleOCR
        for result in results:
            box = result[0]
            top_left = tuple(map(int, box[0]))
            bottom_right = tuple(map(int, box[2]))
            draw.rectangle([top_left, bottom_right], outline="blue", width=3)
    
    return image


# Function to prepare OCR results for CSV download
def prepare_csv_data(easyocr_results, paddleocr_results):

    """
    Prepares OCR results for CSV download.
    Args:
        easyocr_results: Results from EasyOCR.
        paddleocr_results: Results from PaddleOCR.

    Returns:
        A DataFrame containing OCR results.
    """

    data = []
    
    # Extract EasyOCR results
    for easy_result in easyocr_results:
        easy_text = easy_result[1]
        easy_conf = f"{easy_result[2] * 100:.1f}%"
        data.append({"OCR_Type": "EasyOCR", "Text": easy_text, "Confidence": easy_conf})
    
    # Extract PaddleOCR results
    for paddle_result in paddleocr_results[0]:
        try:
            paddle_text = paddle_result[1][0]
            paddle_conf = f"{paddle_result[1][1] * 100:.1f}%"
            data.append({"OCR_Type": "PaddleOCR", "Text": paddle_text, "Confidence": paddle_conf})
        except IndexError:
            data.append({"OCR_Type": "PaddleOCR", "Text": "No text detected", "Confidence": "N/A"})
    
    # Create DataFrame
    df = pd.DataFrame(data)
    return df


# Function to allow downloading CSV
def download_csv(df):
    # Convert DataFrame to CSV in-memory
    csv = df.to_csv(index=False)
    return csv


# Function to compare average confidence between EasyOCR and PaddleOCR
def compare_ocr_performance(easyocr_results, paddleocr_results):

    """
    Compares the average OCR confidence between EasyOCR and PaddleOCR.
    Args:
        easyocr_results: EasyOCR results.
        paddleocr_results: PaddleOCR results.

    Returns:
        Average confidence levels and best OCR method.

    """
    easyocr_confidences = [easy_result[2] for easy_result in easyocr_results]
    paddleocr_confidences = [paddle_result[1][1] for paddle_result in paddleocr_results[0] if len(paddle_result[1]) > 1]
    
    avg_easyocr_confidence = np.mean(easyocr_confidences) * 100 if easyocr_confidences else 0
    avg_paddleocr_confidence = np.mean(paddleocr_confidences) * 100 if paddleocr_confidences else 0
    
    # Determine which OCR performed better
    if avg_easyocr_confidence > avg_paddleocr_confidence:
        best_ocr = "EasyOCR"
    elif avg_paddleocr_confidence > avg_easyocr_confidence:
        best_ocr = "PaddleOCR"
    else:
        best_ocr = "Both are equally effective"
    
    return avg_easyocr_confidence, avg_paddleocr_confidence, best_ocr


# Display recognition details in a box with two columns for EasyOCR and PaddleOCR
def display_recognition_details(easyocr_results, paddleocr_results, best_ocr,avg_easyocr_confidence,avg_paddleocr_confidence):

    """
    # Function to display OCR recognition details.
    # This function shows EasyOCR and PaddleOCR results in a two-column format with their respective confidence levels.

    """
    # Header for recognition details
    st.markdown("<h1 style='text-align: center;color: #008000;'>Recognition Details</h1>", unsafe_allow_html=True)
    
    # Create a container to hold the columns for EasyOCR and PaddleOCR results
    recognition_box = st.container()
    with recognition_box:
        # Create two columns
        col1, col2 = st.columns(2)

        # Display EasyOCR results in the first column
        with col1:
            st.markdown("<h3>EasyOCR</h3>", unsafe_allow_html=True)
            for easy_result in easyocr_results:
                easy_text = html.escape(easy_result[1])  # Escape EasyOCR text
                easy_conf = f"{easy_result[2] * 100:.1f}%"
                st.markdown(f"**{easy_text}** ({easy_conf})", unsafe_allow_html=True)
        
        # Display PaddleOCR results in the second column
        with col2:
            st.markdown("<h3>PaddleOCR</h3>", unsafe_allow_html=True)
            for idx, paddle_result in enumerate(paddleocr_results[0]):
                try:
                    paddle_text = html.escape(paddle_result[1][0])  # Escape PaddleOCR text
                    paddle_conf = f"{paddle_result[1][1] * 100:.1f}%"
                    st.markdown(f"**{paddle_text}** ({paddle_conf})", unsafe_allow_html=True)
                except IndexError:
                    st.markdown("No text detected")

    # Display best OCR and average confidence levels
    st.markdown(f"<h2 style='color: #008000;'>Best OCR: {best_ocr}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #008000;'>Easyocr_confidence: {avg_easyocr_confidence}</h2>", unsafe_allow_html=True)
    st.markdown(f"<h2 style='color: #008000;'>Paddleocr_confidence: {avg_paddleocr_confidence}</h2>", unsafe_allow_html=True)

# Main process
if uploaded_file:
    file_type = uploaded_file.type

    # Check the type of uploaded file and load it accordingly
    if file_type.startswith("image"):
        img = Image.open(uploaded_file)
    elif file_type.endswith("pdf"):
        images = convert_pdf_to_images(uploaded_file)
        img = images[0]  # Use the first page of the PDF as an image

    # Display the uploaded image
    st.image(img, caption="Original Image", use_container_width=True)

    st.markdown("<h1 style='text-align: center; color: #800080;'>DETECTION AND RECOGNITION</h1>", unsafe_allow_html=True)

    if st.button("LAUNCH"):
        # EasyOCR Detection
        with st.spinner("Processing with EasyOCR..."):
            easyocr_results = perform_easyocr(img.copy())
            easyocr_image = draw_boxes(img.copy(), easyocr_results, ocr_type="easyocr")

        # PaddleOCR Detection
        with st.spinner("Processing with PaddleOCR..."):
            paddleocr_results = perform_paddleocr(img.copy())
            paddleocr_image = draw_boxes(img.copy(), paddleocr_results[0], ocr_type="paddleocr")

        st.markdown("<h1 style='text-align: center; color: #008000;'>Images with Detected Text</h1>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.image(easyocr_image, caption="EasyOCR Results", use_container_width=True)
        with col2:
            st.image(paddleocr_image, caption="PaddleOCR Results", use_container_width=True)

        # Compare OCR performances
        avg_easyocr_confidence, avg_paddleocr_confidence, best_ocr = compare_ocr_performance(easyocr_results, paddleocr_results)

        # Display comparison and recognition details
        st.markdown(f"**EasyOCR Average Confidence**: {avg_easyocr_confidence:.1f}%")
        st.markdown(f"**PaddleOCR Average Confidence**: {avg_paddleocr_confidence:.1f}%")
        display_recognition_details(easyocr_results, paddleocr_results, best_ocr,avg_easyocr_confidence,avg_paddleocr_confidence)

        # Prepare and provide download options for CSV
        df = prepare_csv_data(easyocr_results, paddleocr_results)
        csv = download_csv(df)
        st.download_button("Download OCR Results as CSV", csv, "ocr_results.csv", "text/csv")
