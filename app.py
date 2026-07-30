import streamlit as st
from ultralytics import YOLO
import cv2
import tempfile
from pathlib import Path
import anthropic

# Page config
st.set_page_config(page_title="Traffic Detection & Analysis", layout="wide")

st.title("🚗 Traffic Vehicle Detection & NLP Insights")
st.write("Upload a video or image to detect vehicles and generate traffic insights.")

# Load model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Your trained model

model = load_model()

# Class names
class_names = {0: 'car', 1: 'bus', 2: 'truck', 3: 'motorcycle', 4: 'auto_rickshaw'}

# Streamlit UI
uploaded_file = st.file_uploader("Upload video or image", type=['mp4', 'avi', 'jpg', 'png'])

if uploaded_file:
    st.success("✓ File uploaded successfully!")
    
    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as f:
        f.write(uploaded_file.read())
        temp_path = f.name
    
    # Check file type
    file_type = Path(uploaded_file.name).suffix.lower()
    
    if file_type in ['.jpg', '.png']:
        # Image detection
        st.subheader("Detection Results")
        
        # Run inference
        results = model.predict(source=temp_path, conf=0.5)
        
        # Display image
        for result in results:
            im_array = result.plot()
            st.image(im_array, caption="Detection Result", use_column_width=True)
            
            # Count detections
            detections = result.boxes
            class_counts = {}
            for box in detections:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
            # Display metrics
            st.subheader("Detection Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Vehicles", len(detections))
            with col2:
                st.metric("Unique Classes", len(class_counts))
            with col3:
                st.metric("Confidence", "0.5 (threshold)")
            
            # Display class breakdown
            st.subheader("Vehicles Detected")
            for class_name, count in sorted(class_counts.items()):
                st.write(f"  • {class_name.capitalize()}: {count}")
            
            # Generate NLP insight
            st.subheader("AI-Generated Insights")
            
            insight_text = f"""
            Detected {len(detections)} vehicles.
            Breakdown: {', '.join([f'{count} {name}' for name, count in class_counts.items()])}.
            Traffic density: {'Low' if len(detections) < 5 else 'Moderate' if len(detections) < 15 else 'High'}.
            """
            
            st.info(insight_text)
    
    elif file_type in ['.mp4', '.avi']:
        # Video detection
        st.subheader("Processing Video...")
        
        # Show progress
        progress_bar = st.progress(0)
        
        # Run inference on video
        results = model.predict(source=temp_path, conf=0.5, save=True)
        
        # Video info
        cap = cv2.VideoCapture(temp_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        
        st.success(f"✓ Video processed! Total frames: {total_frames}")
        
        # Aggregate detections across video
        total_detections = 0
        class_counts = {}
        
        for result in results:
            total_detections += len(result.boxes)
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = class_names[class_id]
                class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Display metrics
        st.subheader("Video Analysis Metrics")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Detections", total_detections)
        with col2:
            st.metric("Total Frames", total_frames)
        with col3:
            avg_per_frame = round(total_detections / total_frames, 2) if total_frames > 0 else 0
            st.metric("Avg Detections/Frame", avg_per_frame)
        
        # Display class breakdown
        st.subheader("Vehicle Distribution")
        for class_name, count in sorted(class_counts.items()):
            st.write(f"  • {class_name.capitalize()}: {count} ({round(count/total_detections*100, 1)}%)")
        
        # Generate NLP insight
        st.subheader("AI-Generated Insights")
        
        insight_text = f"""
        Video Analysis Summary:
        - Total vehicles detected: {total_detections} across {total_frames} frames
        - Average detection rate: {round(total_detections/total_frames, 2)} per frame
        - Vehicle distribution: {', '.join([f'{count} {name}' for name, count in class_counts.items()])}
        - Busiest class: {max(class_counts, key=class_counts.get) if class_counts else 'None'}
        - Traffic density: {'Low' if total_detections < 100 else 'Moderate' if total_detections < 300 else 'High'}
        """
        
        st.info(insight_text)
    
    # Cleanup
    Path(temp_path).unlink()

# Sidebar info
st.sidebar.header("About")
st.sidebar.write("""
This app uses YOLOv8s to detect vehicles in real-time.

**Classes:** Car, Bus, Truck, Motorcycle, Auto-rickshaw

**Model:** Trained on balanced Indian traffic dataset (mAP50: 0.749)

**Source:** [GitHub](https://github.com/Mukesh-2005/traffic-vehicle-detection)
""")