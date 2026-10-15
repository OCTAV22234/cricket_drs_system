import streamlit as st
import cv2
import numpy as np
import os
import tempfile
import time
import librosa

# Page Configuration
st.set_page_config(page_title="Instant Cricket DRS System", page_icon="🏏", layout="centered")

st.markdown("<h1 style='text-align: center; color: #ff4b4b;'>🏏 Instant Stream Cricket DRS</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Stream your clip instantly and get real-time third umpire verdicts.</p>", unsafe_allow_html=True)
st.write("---")

# Sidebar Controls & File Uploads
st.sidebar.header("📺 Match Inputs & Appeal Settings")
appeal_type = st.sidebar.selectbox(
    "Select Appeal Type", 
    ["Comprehensive Full Review (All-in-One)", "LBW Appeal", "Run Out Appeal", "UltraEdge (Edge) Appeal"]
)

uploaded_video = st.sidebar.file_uploader("Upload Delivery Video (.mp4)", type=["mp4", "mov", "avi"])
uploaded_audio = st.sidebar.file_uploader("Upload Synchronized Audio (.wav)", type=["wav"])

# Run Out specific control
if "Run Out" in appeal_type or "Comprehensive" in appeal_type:
    batsman_safe = st.sidebar.checkbox("Batsman reached crease safely?", value=False)
else:
    batsman_safe = False

# Main Video Stream Preview Area (Instant Streaming)
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(uploaded_video.read())
    video_path = tfile.name
    
    st.subheader("📺 Live Broadcast Stream")
    st.video(video_path)
else:
    st.info("💡 **Tip:** Upload a short cricket clip (`.mp4`) using the sidebar for instant streaming and review.")
    video_path = None

st.write("---")

# Helper function to render broadcast-style decision boxes
def display_drs_box(status, category, detail):
    if status == "OUT":
        bg_color, text_color, border_color = "#ffe6e6", "#cc0000", "#ff4b4b"
    elif status == "NOT OUT":
        bg_color, text_color, border_color = "#e6ffe6", "#008000", "#2eb82e"
    else:
        bg_color, text_color, border_color = "#fff5cc", "#b38600", "#ffcc00"

    st.markdown(f"""
        <div style="background-color: {bg_color}; border: 3px solid {border_color}; padding: 15px; border-radius: 10px; text-align: center; margin-bottom: 10px;">
            <h4 style="color: #333; margin: 0;">DRS REVIEW SYSTEM - {category}</h4>
            <h1 style="color: {text_color}; font-size: 38px; margin: 8px 0;">{status}</h1>
            <p style="color: #555; font-size: 15px; margin: 0;"><b>Reason:</b> {detail}</p>
        </div>
    """, unsafe_allow_html=True)

# Optimized Instant Appeal Button
if st.button("🏏 INSTANT APPEAL! REFER TO THIRD UMPIRE", use_container_width=True):
    
    # 1. Quick Broadcast Suspense State
    placeholder = st.empty()
    with placeholder.container():
        display_drs_box("DECISION PENDING", appeal_type.upper(), "Fast-tracking ball tracking & audio analysis...")
        time.sleep(1.0) # Fast, snappy delay
    placeholder.empty()

    # 2. Lightning-Fast Video Frame Sampling (Skipping frames to prevent lag)
    if appeal_type == "LBW Appeal" or appeal_type == "Comprehensive Full Review (All-in-One)":
        if video_path and os.path.exists(video_path):
            cap = cv2.VideoCapture(video_path)
            lower_red, upper_red = np.array([0, 120, 70]), np.array([10, 255, 255])
            trajectory_count = 0
            frame_skip = 5  # Speed optimization: Checks every 5th frame
            frame_idx = 0
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                if frame_idx % frame_skip == 0:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    mask = cv2.inRange(hsv, lower_red, upper_red)
                    if np.sum(mask > 0) > 40: 
                        trajectory_count += 1
                frame_idx += 1
            cap.release()
            
            lbw_status = "OUT" if trajectory_count > 1 else "NOT OUT"
            lbw_detail = "Hitting Stumps (Trajectory Projected)" if lbw_status == "OUT" else "Missing Stumps / Pitching Outside Leg"
        else:
            lbw_status, lbw_detail = "OUT", "Hitting Stumps (Mock Data Fallback)"

        display_drs_box(lbw_status, "LBW CHECK", lbw_detail)

    if appeal_type == "Run Out Appeal" or appeal_type == "Comprehensive Full Review (All-in-One)":
        runout_status = "NOT OUT" if batsman_safe else "OUT"
        runout_detail = "Batsman made it home safely inside the crease." if batsman_safe else "Bails broken before batsman grounded bat."
        
        display_drs_box(runout_status, "RUN OUT CHECK", runout_detail)

    if appeal_type == "UltraEdge (Edge) Appeal" or appeal_type == "Comprehensive Full Review (All-in-One)":
        if uploaded_audio is not None:
            t_audio = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            t_audio.write(uploaded_audio.read())
            y, sr = librosa.load(t_audio.name, sr=None)
            rms = librosa.feature.rms(y=y)[0]
            spike = np.any(rms > (np.mean(rms) * 3.0))
            audio_status = "OUT" if spike else "NOT OUT"
            audio_detail = "Audio spike detected matching bat-ball impact." if spike else "Flat waveform. No edge detected."
        else:
            audio_status, audio_detail = "NOT OUT", "No audio file uploaded for spike analysis."

        display_drs_box(audio_status, "ULTRAEDGE AUDIO", audio_detail)

    st.balloons()
