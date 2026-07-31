import cv2
import streamlit as st
import numpy as np

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Face Detection App",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Face Detection using OpenCV")
st.write("Upload an image and the app will detect all faces present in it.")

st.divider()

# -----------------------------
# Load Face Detector
# -----------------------------
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_detector = cv2.CascadeClassifier(cascade_path)

if face_detector.empty():
    st.error("❌ Face detector model could not be loaded.")
    st.stop()


# -----------------------------
# Upload Image
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload an Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Convert uploaded file to OpenCV image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    st.subheader("Original Image")
    st.image(
        cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw rectangles
    output = image.copy()

    for (x, y, w, h) in faces:
        cv2.rectangle(
            output,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

    st.subheader("Detected Faces")
    st.image(
        cv2.cvtColor(output, cv2.COLOR_BGR2RGB),
        use_container_width=True
    )

    st.success(f"✅ Faces Detected: {len(faces)}")

else:
    st.info("Please upload an image to begin.")