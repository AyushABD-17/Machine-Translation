import streamlit as st
import requests
import json
from PIL import Image
import base64
import time

st.set_page_config(page_title="Multimodal Neural Machine Translation", layout="wide")

st.title("Multimodal Neural Machine Translation 🌍")
st.write("English to French translation spanning Text, Images, and Audio via our End-to-End Deep Learning Architecture.")

st.sidebar.header("Navigation")
app_mode = st.sidebar.selectbox("Choose Mode", ["Live Camera Translate", "Local Image Translation"])

API_BASE_URL = "http://localhost:8000"

# Auto-play audio script
def autoplay_audio(file_path: str):
    with open(file_path, "rb") as f:
        data = f.read()
        b64 = base64.b64encode(data).decode()
        md = f"""
            <audio controls autoplay="true">
            <source src="data:audio/mp3;base64,{b64}" type="audio/mp3">
            </audio>
            """
        st.markdown(md, unsafe_allow_html=True)

if app_mode == "Live Camera Translate":
    st.header("Camera → OCR → Translation → Voice Pipeline")
    
    if st.button("Capture & Translate 📷"):
        with st.spinner("Capturing image and processing through the multimodal pipeline..."):
            try:
                # Calls the complete integrated pipeline locally behind FastAPI
                response = requests.get(f"{API_BASE_URL}/camera_translate")
                
                if response.status_code == 200:
                    result = response.json()
                    
                    if "error" in result:
                        st.error(f"Error: {result['error']}")
                    else:
                        st.success("Translation Pipeline Completed!")
                        
                        # Show data
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("### Extracted Text (English OCR)")
                            st.info(result.get("detected_text", "No text detected."))
                            
                        with col2:
                            st.write("### Translation (French NMT)")
                            st.success(result.get("translation", "No text to translate."))
                        
                        audio_file = result.get("audio_file")
                        if audio_file:
                            st.write("### Audio Output")
                            autoplay_audio(audio_file)
                            
                        # Show the image for proof 
                        # We know camera pipeline saves to data/captures/capture.jpg
                        try:
                            img = Image.open("data/captures/capture.jpg")
                            st.image(img, caption="Latest Capture", use_container_width=True)
                        except FileNotFoundError:
                            pass
                else:
                    st.error("Failed to connect to the backend translation service. Is FastAPI running?")
            except Exception as e:
                 st.error(f"Failed to connect to API: {e}")

elif app_mode == "Local Image Translation":
    st.header("Upload Image for Translation")
    
    uploaded_file = st.file_uploader("Choose an image containing English text...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        import os
        os.makedirs("data/uploads", exist_ok=True)
        save_path = f"data/uploads/{uploaded_file.name}"
        
        with open(save_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        
        if st.button("Translate Image"):
            with st.spinner("Processing image..."):
                try:
                    payload = {"image_path": save_path}
                    response = requests.post(f"{API_BASE_URL}/translate_image", json=payload)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.write("### Detected Text")
                        st.info(result.get("detected_text", "None"))
                        st.write("### Translation")
                        st.success(result.get("translation", "None"))
                    else:
                        st.error("Failed to get response form server.")
                except Exception as e:
                    st.error(f"Failed to connect to API: {e}")
