# %%writefile app.py
import streamlit as st
import cv2

# Load Haar Cascade Classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(scale_factor, min_neighbors, rect_color_hex):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        return

    stframe = st.empty()

    # Convert hex to BGR color
    hex_color = rect_color_hex.lstrip('#')
    bgr_color = tuple(int(hex_color[i:i+2], 16) for i in (4, 2, 0))  # Convert R,G,B to B,G,R

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Error: Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors
        )

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), bgr_color, 2)

        stframe.image(frame, channels="BGR")

    cap.release()

def app():
    st.title("📸 Face Detection using Viola-Jones Algorithm")
    st.write("Adjust the settings below and click **Detect Faces** to use your webcam.")

    # Add Streamlit controls
    rect_color = st.color_picker("Pick rectangle color", "#00FF00")
    scale_factor = st.slider("Adjust scaleFactor", min_value=1.05, max_value=2.0, value=1.3, step=0.05)
    min_neighbors = st.slider("Adjust minNeighbors", min_value=1, max_value=10, value=5)

    if st.button("Detect Faces"):
        detect_faces(scale_factor, min_neighbors, rect_color)

if __name__ == "__main__":

