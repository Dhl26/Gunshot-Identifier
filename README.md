# Audio Forensics: Gunshot Identifier 🔫

An advanced AI-powered forensic tool for identifying firearm types from audio recordings. This project utilizes deep Convolutional Neural Networks (CNNs) and Mel-spectrogram analysis to classify gunshot sounds with high accuracy, robust to environmental noise.

## 🌟 Features

*   **Multi-Architecture Support**: Train and use various state-of-the-art CNN architectures:
    *   ResNet (18, 50, 101)
    *   GoogLeNet
    *   Inception-v3
    *   InceptionResNetV2
*   **Web Interface**: A modern, sleek Streamlit web app for easy interaction and visualization.
*   **Visual Forensics**: View Mel-spectrograms and probability distributions for every analysis.
*   ** robust Preprocessing**: Automatic noise injection, time-shifting, and resizing to handle diverse audio inputs.

## 🛠️ Installation

1.  **Clone the repository** (or ensure you are in the project directory).
2.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: You may need to install `torch` specifically for your CUDA version if you plan to use GPU acceleration. Visit [pytorch.org](https://pytorch.org/).*

## 🚀 Usage

### 1. Training a Model

You can train a new detector using the provided script. The default model is `resnet50`.

```bash
# Train with default ResNet50
python gunshot_identifier.py --train

# Train with a specific architecture
python gunshot_identifier.py --train --model inception_v3
```

**Supported Models (`--model`):**
`resnet18`, `resnet50`, `resnet101`, `googlenet`, `inception_v3`, `inceptionresnetv2`

The training script will:
1. Scan the `firearm sounds refined` directory (or directory specified by `--data_dir`).
2. Balance the dataset using oversampling.
3. Train the chosen CNN architecture.
4. Save the model to `gunshot_cnn_model.pth` and the label encoder to `cnn_label_encoder.pkl`.

### 2. Running the Web App

Launch the interactive dashboard to use the trained model:

```bash
streamlit run app.py
```

*   Upload a `.wav` file.
*   The app automatically loads the correct model architecture.
*   View the predicted weapon and confidence score.

### 3. CLI Prediction

You can also run predictions purely from the command line:

```bash
python gunshot_identifier.py --predict path/to/audio.wav
```

## 📂 Project Structure

*   `app.py`: Streamlit web application.
*   `gunshot_identifier.py`: Core logic for dataset handling, training, and inference.
*   `requirements.txt`: Python dependencies.
*   `gunshot_cnn_model.pth`: Saved PyTorch model (generated after training).
*   `cnn_label_encoder.pkl`: Saved Label Encoder (generated after training).
