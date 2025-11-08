import streamlit as st
import numpy as np
import pandas as pd
import pickle
import os
import gdown

# ------------------------------------------------------------
# 📁 STEP 1 — Create local folder for model/data
# ------------------------------------------------------------
MODEL_DIR = "beats_model"
os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------------------------------------------
# ☁️ STEP 2 — Download model and data files from Google Drive
# ------------------------------------------------------------
files = {
    "best_model.pkl": "1Lx3eLuI0HMzeyCKcm7v8CEsvsuuiC7jk",  # ✅ your model
    "train.csv": "1TeKaqSNmAe0yScnb6UpdnfekH3C1d0ya",
    "test.csv": "1s2uI4slJCfSC9I6OO0ETwX8z-4b13bxe"
}

for name, fid in files.items():
    dest_path = os.path.join(MODEL_DIR, name)
    if not os.path.exists(dest_path):
        st.write(f"📥 Downloading {name} from Google Drive...")
        gdown.download(f"https://drive.google.com/uc?id={fid}", dest_path, quiet=False)

# ------------------------------------------------------------
# 🧠 STEP 3 — Cached model loading (to speed up app reloads)
# ------------------------------------------------------------
@st.cache_resource
def load_model():
    """Load and cache the ML model to avoid reloading every time."""
    with open(os.path.join(MODEL_DIR, "best_model.pkl"), "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

# SETTING PAGE TITLE 
st.set_page_config(page_title=" Music BPM Predictor",page_icon="🎧",layout='wide')

# SIDEBAR 
st.sidebar.title(" Navigation")
page = st.sidebar.radio("Go to",["Home","Single Prediction","Batch Prediction"])

if page == "Home":
    st.title(" Music BPM Predictor")
    st.write("""
    Welcome to the Music BPM Predictor app! ...
    
    """)
    st.image("https://images.unsplash.com/photo-1487215078519-e21cc028cb29?auto=format&fit=crop&w=1200&q=60")

elif page == 'Single Prediction':
    st.title("Single Song BPM Prediction")
    
    # INPUT COLUMNS 
    col1 , col2, col3 = st.columns(3)
    
    with col1:
        RhythmScore = st.slider("Rhythm Score",0.0,1.0,0.1,step = 0.01)
        AudioLoudness = st.slider("Audio Loudness",-27.0,-1.0,-20.0,step = 0.1)
        VocalContent = st.slider('Vocal Content',0.0,0.25,0.1,step=0.01)
    with col2:
        AcousticQuality = st.number_input("Acoustic Quality",0.000005,1.0,0.5,step=0.01)
        InstrumentalScore = st.slider("Instrumental Score",0.000001,0.87,0.5,step=0.01)
        LivePerformanceLikelihood = st.number_input("Live Performance Likelihood",0.02,0.6,0.5,step=0.01)
    with col3:
        MoodScore = st.slider("Mood Score",0.02,0.98,0.5,step=0.01)
        TrackDurationMs = st.number_input('Track Duration (ms)',60000,465000,210000,step = 500)
        TrackDurationMs = np.ceil(TrackDurationMs/60000)
        Energy = st.number_input("Energy",0.0,1.0,0.5,step=0.05)
        
    if st.button("🎵 Predict BPM"):
        input_data = pd.DataFrame({
            'RhythmScore':[RhythmScore],
            'AudioLoudness':[AudioLoudness],
            'VocalContent':[VocalContent],
            'AcousticQuality':[AcousticQuality],
            'InstrumentalScore':[InstrumentalScore],
            'LivePerformanceLikelihood':[LivePerformanceLikelihood],
            'MoodScore':[MoodScore],
            'TrackDurationMs':[TrackDurationMs],
            'Energy':[Energy]
        })
        
        prediction = model.predict(input_data)
        st.success(f"Predicted Beats Per Minute (BPM): {prediction[0]:.2f}")
    
elif page == 'Batch Prediction':
    st.title("Batch BPM Prediction")
    st.write("Upload a CSV file containing song features to predict BPM for multiple songs.")
    
    @st.cache_data
    def load_csv(uploaded_file):
        """Cache uploaded CSV file reading."""
        return pd.read_csv(uploaded_file)
    
    uploaded_file = st.file_uploader('Upload your csv file ', type=['csv'])
    
    if uploaded_file is not None:
        df = load_csv(uploaded_file)
        st.write("### Preview of Uploaded CSV File")
        st.dataframe(df)
        
        if st.button("🎵 Predict BPM for Batch"):
            #  Drop 'id' column 
            if 'id' in df.columns:
                df = df.drop(columns=['id'])
            
            #  Make predictions
            prediction = model.predict(df)
            df['Prediction_BPM'] = prediction
            
            st.success(" Predictions Completed!")
            st.dataframe(df.head())
            
           
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button(" Download Predictions", csv, "bpm_predictions.csv", "text/csv")
