import streamlit as st
import torch
import librosa
import numpy as np
import joblib
import matplotlib.pyplot as plt
import os
import sys

# Add current directory to path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gunshot_identifier import GunshotCNN, MODEL_PATH, LABEL_ENCODER_PATH, SAMPLE_RATE, N_MELS, SAMPLES_PER_TRACK, get_input_size

# --- Configuration & Styling ---
st.set_page_config(
    page_title="Audio Forensics | Gunshot Classifier",
    page_icon="🔫",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for "Rich Aesthetics"
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        color: #ffffff;
    }
    .stButton>button {
        background: linear-gradient(90deg, #FF4B4B 0%, #FF9056 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 8px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 75, 75, 0.4);
    }
    .prediction-card {
        background-color: #1a1c24;
        padding: 2rem;
        border-radius: 12px;
        border: 1px solid #2d2f36;
        text-align: center;
        margin-bottom: 2rem;
    }
    .highlight-text {
        color: #FF4B4B;
        font-size: 2.5rem;
        font-weight: 800;
    }
    .confidence-text {
        color: #a0a0a0;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_resource
def load_resources():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        return None, None, None, None

    # Load Label Encoder
    le = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(le.classes_)
    
    # Load Checkpoint with Architecture Check
    # AUTOMATIC JOIN: If model is missing but parts exist, join them
    if not os.path.exists(MODEL_PATH):
        import glob
        parts = glob.glob(f"{MODEL_PATH}.part*")
        if parts:
            st.info("Reconstructing model from parts... this may take a moment.")
            from model_manager import join_model
            join_model(MODEL_PATH)
            st.success("Model reconstructed successfully!")

    try:
        checkpoint = torch.load(MODEL_PATH, map_location=device)
        model_name = "custom"
        state_dict = checkpoint
        
        if isinstance(checkpoint, dict) and "model_name" in checkpoint:
            model_name = checkpoint["model_name"]
            state_dict = checkpoint["state_dict"]
            
        # Load Model
        model = GunshotCNN(num_classes, model_name).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        target_size = get_input_size(model_name)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None, None
    
    return model, le, device, target_size

def process_audio(audio_file):
    # Load audio
    audio, _ = librosa.load(audio_file, sr=SAMPLE_RATE)
    
    # Pad or Truncate
    if len(audio) > SAMPLES_PER_TRACK:
        audio = audio[:SAMPLES_PER_TRACK]
    else:
        padding = SAMPLES_PER_TRACK - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
        
    # Spectrogram
    mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
    mel_spectrogram_db = librosa.power_to_db(mel_spectrogram, ref=np.max)
    
    return audio, mel_spectrogram_db

# --- Main App ---
def main():
    st.title("🔫 Audio Forensics: Gunshot Identifier")
    st.markdown("Upload a gunshot audio recording to identify the weapon type using our advanced CNN model.")
    
    # About Info (Moved from Sidebar)
   
    
    # Model Loading
    model, le, device, target_size = load_resources()
    
    if model is None:
        st.error(f"Model not found at `{MODEL_PATH}`. Please run the training script first!")
        return
        

    # File Upload
    uploaded_file = st.file_uploader("Choose a WAV file", type=["wav"])

    if uploaded_file is not None:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.audio(uploaded_file, format='audio/wav')
            st.write("Processing audio...")
            
            try:
                # Preprocess
                audio_data, spec_db = process_audio(uploaded_file)
                
                # Resize and Predict
                # (1, 1, H, W)
                spec_tensor = torch.tensor(spec_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                
                # Resize to model target size
                spec_resized = torch.nn.functional.interpolate(
                    spec_tensor, size=target_size, mode='bilinear', align_corners=False
                ).to(device)
                
                with torch.no_grad():
                    outputs = model(spec_resized)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    confidence, predicted_idx = torch.max(probabilities, 1)
                
                predicted_label = le.inverse_transform([predicted_idx.item()])[0]
                conf_score = confidence.item() * 100
                
                # --- Metrics ---
                st.markdown("### Analysis Results")
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div class="confidence-text">Identified Weapon</div>
                    <div class="highlight-text">{predicted_label}</div>
                    <div class="confidence-text">Confidence: {conf_score:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.stop()

        with col2:
            st.subheader("Visual Analysis")
            
            # Probability Chart
            st.markdown("**Probability Distribution**")
            probs_np = probabilities.cpu().numpy()[0]
            classes = le.classes_
            
            import pandas as pd
            df_probs = pd.DataFrame({
                'Weapon': classes,
                'Probability': probs_np
            }).sort_values(by='Probability', ascending=True)
            
            st.bar_chart(df_probs.set_index('Weapon'), color="#FF4B4B")
            
            # Spectrogram Visualization
            st.markdown("**Mel-Spectrogram**")
            fig, ax = plt.subplots(figsize=(10, 4))
            img = librosa.display.specshow(spec_db, x_axis='time', y_axis='mel', sr=SAMPLE_RATE, ax=ax, cmap='magma')
            fig.colorbar(img, ax=ax, format='%+2.0f dB')
            ax.set_title('Mel-frequency Spectrogram')
            # Dark theme for matplotlib plot
            fig.patch.set_facecolor('#0e1117')
            ax.set_facecolor('#0e1117')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.title.set_color('white')
            
            st.pyplot(fig)

if __name__ == "__main__":
    main()
