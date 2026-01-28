import os
import glob
import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import logging
import argparse
import sys
try:
    import timm
except ImportError:
    timm = None
import torchvision.models as models
from torchvision import transforms

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
DATA_DIR = r"d:/NFSU/Research/gunshots/firearm sounds refined"
MODEL_PATH = "gunshot_cnn_model.pth"
LABEL_ENCODER_PATH = "cnn_label_encoder.pkl"

# Audio config
SAMPLE_RATE = 22050
DURATION = 2.0  # Seconds per window (Standardized)
SAMPLES_PER_TRACK = int(SAMPLE_RATE * DURATION)
N_MELS = 64     # Height of spectrogram

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_input_size(model_name):
    if "inception" in model_name.lower():
        return (299, 299)
    return (224, 224)

class GunshotDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None, augment=False, target_size=(224, 224)):
        self.file_paths = file_paths
        self.labels = labels
        self.transform = transform
        self.augment = augment
        self.target_size = target_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load audio
        try:
            audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        except Exception as e:
            logging.error(f"Error loading {file_path}: {e}")
            return torch.zeros((1, N_MELS, int(SAMPLES_PER_TRACK / 512) + 1)), torch.tensor(label, dtype=torch.long)

        # Pad or Truncate
        if len(audio) > SAMPLES_PER_TRACK:
            audio = audio[:SAMPLES_PER_TRACK]
        else:
            padding = SAMPLES_PER_TRACK - len(audio)
            audio = np.pad(audio, (0, padding), mode='constant')

        # Augmentation (Noise Injection & Time Shift)
        if self.augment:
            audio = self._augment_audio(audio)

        # Feature Extraction: Mel Spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Add channel dimension for CNN: (1, n_mels, time_steps)
        spec_tensor = torch.tensor(mel_spectrogram, dtype=torch.float32).unsqueeze(0)
        
        # Resize to target size for pretrained models
        spec_tensor = torch.nn.functional.interpolate(
            spec_tensor.unsqueeze(0), size=self.target_size, mode='bilinear', align_corners=False
        ).squeeze(0)
        
        return spec_tensor, torch.tensor(label, dtype=torch.long)

    def _augment_audio(self, audio):
        # 1. Add Gaussian Noise
        if np.random.rand() < 0.5:
            noise_amp = 0.05 * np.random.uniform() * np.amax(audio)
            audio = audio + noise_amp * np.random.normal(size=audio.shape[0])
        
        # 2. Time Shift
        if np.random.rand() < 0.5:
            shift = int(np.random.uniform(low=-0.2, high=0.2) * len(audio))
            audio = np.roll(audio, shift)
            
        # 3. Pitch Shift (New)
        if np.random.rand() < 0.5:
            try:
                # Shift by -2 to +2 semitones
                n_steps = np.random.uniform(-2, 2)
                audio = librosa.effects.pitch_shift(audio, sr=SAMPLE_RATE, n_steps=n_steps)
            except Exception:
                pass # Fallback if audio is too short or error

        # 4. Time Stretch (New)
        if np.random.rand() < 0.5:
             try:
                rate = np.random.uniform(0.8, 1.2)
                audio = librosa.effects.time_stretch(audio, rate=rate)
                # Stretching changes length, strict length enforcement handles this later in __getitem__ but
                # we are inside the method called before feature extraction?
                # Wait, __getitem__ calls _augment_audio AFTER padding.
                # Time stretch changes length. We need to re-pad/crop inside here or handle it.
                # Let's fix length here to match input length.
                target_len = SAMPLES_PER_TRACK
                if len(audio) > target_len:
                    audio = audio[:target_len]
                else:
                    audio = np.pad(audio, (0, target_len - len(audio)), mode='constant')
             except Exception:
                pass
            
        return audio

class GunshotCNN(nn.Module):
    def __init__(self, num_classes, model_name="resnet50"):
        super(GunshotCNN, self).__init__()
        self.model_name = model_name.lower()
        self.aux_logits = False
        
        if self.model_name == "custom":
            self.model = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
                nn.Flatten(),
                nn.AdaptiveAvgPool2d((4, 4)), # Note: Flatten happens before this in original, but logic was mixed.
                # Re-implementing original logic correctly using sequential if needed, 
                # but original had pool/flatten interspersed.
                # Let's keep it simple: Define custom locally if needed or just use consistent block.
            )
            # Reverting custom block structure to avoid complexity, using simple fallback:
            self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
            self.relu = nn.ReLU()
            self.pool = nn.MaxPool2d(2)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
            self.global_pool = nn.AdaptiveAvgPool2d((4, 4))
            self.fc1 = nn.Linear(64 * 4 * 4, 128)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(128, num_classes)
            
        elif self.model_name in ["resnet18", "resnet50", "resnet101"]:
            backbone = getattr(models, self.model_name)(pretrained=True)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, num_classes)
            self.model = backbone
            
        elif self.model_name == "googlenet":
            backbone = models.googlenet(pretrained=True)
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, num_classes)
            self.model = backbone
            
        elif self.model_name == "inception_v3":
            backbone = models.inception_v3(pretrained=True)
            self.aux_logits = True
            # Handle auxiliary net
            num_ftrs = backbone.AuxLogits.fc.in_features
            backbone.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle primary net
            num_ftrs = backbone.fc.in_features
            backbone.fc = nn.Linear(num_ftrs, num_classes)
            self.model = backbone
            
        elif self.model_name == "inceptionresnetv2":
            if timm:
                self.model = timm.create_model('inception_resnet_v2', pretrained=True, num_classes=num_classes)
            else:
                raise ImportError("timm library needed for InceptionResNetV2")
        else:
            raise ValueError(f"Unknown model name: {model_name}")

    def forward(self, x):
        if self.model_name == "custom":
            x = self.pool(self.relu(self.conv1(x)))
            x = self.pool(self.relu(self.conv2(x)))
            x = self.pool(self.relu(self.conv3(x)))
            x = self.global_pool(x)
            x = x.view(x.size(0), -1) # Flatten
            x = self.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x
        
        # Pretrained models expect 3 channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
            
        if self.model_name == "inception_v3" and self.training:
            return self.model(x) # Returns (logits, aux_logits)
            
        return self.model(x)

def get_label_from_filename(filename):
    """
    GT Label extraction.
    """
    filename_lower = filename.lower()
    if "22 rifle" in filename_lower: return "22 Rifle"
    elif "carbine" in filename_lower: return "Carbine"
    elif "ak" in filename_lower: return "AK-47"
    elif "insas" in filename_lower: return "INSAS"
    elif "pistol" in filename_lower: return "Pistol"
    else: return "Unknown"

def prepare_data():
    files = glob.glob(os.path.join(DATA_DIR, "*.wav"))
    valid_files = []
    labels = []
    
    print(f"Scanning directory for training data: {DATA_DIR}")
    for f in files:
        label = get_label_from_filename(os.path.basename(f))
        if label != "Unknown":
            valid_files.append(f)
            labels.append(label)
    
    print(f"Found {len(valid_files)} valid audio samples.")
    le = LabelEncoder()
    labels_encoded = le.fit_transform(labels)
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        valid_files, labels_encoded, test_size=0.2, random_state=42, stratify=labels_encoded
    )
    
    # --- OVERSAMPLING LOGIC ---
    # Since some classes (like Carbine) have very few samples (e.g., 2), we MUST oversample them
    # in the training set to prevent the model from ignoring them.
    
    unique_classes, class_counts_train = np.unique(train_labels, return_counts=True)
    max_count = np.max(class_counts_train)
    
    print(f"\nOriginal Train Distribution: {dict(zip(le.inverse_transform(unique_classes), class_counts_train))}")
    print(f"Oversampling minority classes to ~{max_count} samples each...")
    
    train_files_resampled = list(train_files)
    train_labels_resampled = list(train_labels)
    
    for cls in unique_classes:
        cls_indices = [i for i, x in enumerate(train_labels) if x == cls]
        current_count = len(cls_indices)
        
        if current_count < max_count:
            # Calculate shortage
            diff = max_count - current_count
            # Randomly sample existing indices to duplicate
            if current_count > 0:
                add_indices = np.random.choice(cls_indices, diff, replace=True)
                for idx in add_indices:
                    train_files_resampled.append(train_files[idx])
                    train_labels_resampled.append(train_labels[idx])
            else:
                 print(f"Warning: Class {cls} has 0 training samples! Cannot oversample.")

    train_files = np.array(train_files_resampled)
    train_labels = np.array(train_labels_resampled)
    
    # Recalculate counts after resampling
    final_class_counts = np.bincount(train_labels)
    total_samples = len(train_labels)
    
    # Weights should now be roughly 1.0 since data is balanced, but we keep the logic just in case
    num_classes = len(le.classes_)
    class_weights = total_samples / (num_classes * final_class_counts)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    
    print("\nFinal Balanced Distribution:")
    for idx, count in enumerate(final_class_counts):
        # Handle case where a class might be missing from training entirely
        if idx < len(le.classes_):
             print(f"  {le.classes_[idx]}: {count} samples")
    
    return train_files, val_files, train_labels, val_labels, le, class_weights

def train_model(model_name="resnet50"):
    train_files, val_files, train_labels, val_labels, le, class_weights = prepare_data()
    
    target_size = get_input_size(model_name)
    print(f"Using model: {model_name} | Input Size: {target_size}")
    
    train_dataset = GunshotDataset(train_files, train_labels, augment=True, target_size=target_size)
    val_dataset = GunshotDataset(val_files, val_labels, augment=False, target_size=target_size)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True) # Increased batch size slightly
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    
    num_classes = len(le.classes_)
    print(f"Classes: {le.classes_}")
    
    model = GunshotCNN(num_classes, model_name).to(device)
    
    # Use weighted loss
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = optim.Adam(model.parameters(), lr=0.0001) # Lower LR for finetuning
    
    joblib.dump(le, LABEL_ENCODER_PATH)
    
    num_epochs = 30
    print(f"Starting training for {num_epochs} epochs on {device}...")
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            
            if model_name == "inception_v3":
                outputs, aux_outputs = model(inputs)
                loss1 = criterion(outputs, targets)
                loss2 = criterion(aux_outputs, targets)
                loss = loss1 + 0.4 * loss2
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if model_name == "inception_v3":
                # For acc calculation use main outputs
                _, predicted = torch.max(outputs.data, 1)
            else:
                _, predicted = torch.max(outputs.data, 1)
                
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
            
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        
        if (epoch + 1) % 5 == 0:
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += targets.size(0)
                    val_correct += (predicted == targets).sum().item()
            
            val_acc = 100 * val_correct / val_total
            print(f"Epoch [{epoch+1}/{num_epochs}] Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Val Acc: {val_acc:.2f}%")
        
    # Save model and architecture info
    save_data = {
        'state_dict': model.state_dict(),
        'model_name': model_name,
        'num_classes': num_classes
    }
    torch.save(save_data, MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

def predict(file_path):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(LABEL_ENCODER_PATH):
        print("Model not found. Please train first using --train.")
        return

    le = joblib.load(LABEL_ENCODER_PATH)
    num_classes = len(le.classes_)
    
    # Load model with architecture check
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_name' in checkpoint:
        model_name = checkpoint['model_name']
        print(f"Loading {model_name} architecture...")
    else:
        # Legacy fallback
        model_name = "custom"
        print("Loading legacy custom architecture...")
        
    model = GunshotCNN(num_classes, model_name).to(device)
    
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) # Legacy
        
    model.eval() 
    
    target_size = get_input_size(model_name)
    
    # Load audio
    try:
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    # SLIDING WINDOW PREDICTION
    # We scan the audio in steps to capture the gunshot wherever it is.
    
    step = int(SAMPLES_PER_TRACK * 0.5) # 50% overlap
    windows = []
    
    # If audio is shorter than window, just pad it
    if len(audio) <= SAMPLES_PER_TRACK:
        padding = SAMPLES_PER_TRACK - len(audio)
        window = np.pad(audio, (0, padding), mode='constant')
        windows.append(window)
    else:
        # Generate windows
        for i in range(0, len(audio) - int(SAMPLES_PER_TRACK * 0.5), step):
            window = audio[i : i + SAMPLES_PER_TRACK]
            # Pad the last window if needed
            if len(window) < SAMPLES_PER_TRACK:
                padding = SAMPLES_PER_TRACK - len(window)
                window = np.pad(window, (0, padding), mode='constant')
            
            # Ensure strict length
            if len(window) == SAMPLES_PER_TRACK:
                windows.append(window)
    
    if not windows:
        # Fallback for edge cases
        windows.append(np.pad(audio[:SAMPLES_PER_TRACK], (0, max(0, SAMPLES_PER_TRACK - len(audio))), mode='constant'))

    # Process all windows
    batch_specs = []
    for win in windows:
        mel = librosa.feature.melspectrogram(y=win, sr=SAMPLE_RATE, n_mels=N_MELS)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Resize logic for prediction matching training
        spec_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
        spec_resized = torch.nn.functional.interpolate(
            spec_tensor, size=target_size, mode='bilinear', align_corners=False
        ).squeeze(0).squeeze(0).numpy()
        
        batch_specs.append(spec_resized)
    
    # (Batch_Size, 1, n_mels, time)
    batch_tensor = torch.tensor(np.array(batch_specs), dtype=torch.float32).unsqueeze(1).to(device)
    
    print(f"Scanning {len(windows)} segments of {DURATION}s each...")
    
    with torch.no_grad():
        outputs = model(batch_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        
        # MEAN POOLING: Average the probabilities across all time windows
        avg_probs = torch.mean(probabilities, dim=0)
        confidence, predicted_idx = torch.max(avg_probs, 0)
        
    predicted_label = le.inverse_transform([predicted_idx.item()])[0]
    
    print(f"\nPrediction for {os.path.basename(file_path)}:")
    print(f"Identified Gun: {predicted_label}")
    print(f"Confidence: {confidence.item()*100:.2f}%")
    
    print("Probabilities (Averaged):")
    probs_np = avg_probs.cpu().numpy()
    for idx, prob in enumerate(probs_np):
        print(f"  {le.classes_[idx]}: {prob*100:.2f}%")

if __name__ == "__main__":
    print("--- Gunshot CNN Classifier ---")
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--predict", type=str, help="Path to wav file")
    parser.add_argument("--data_dir", type=str, help="Path to training data directory")
    parser.add_argument("--model", type=str, default="resnet50", 
                        choices=['resnet18', 'resnet50', 'resnet101', 'googlenet', 'inception_v3', 'inceptionresnetv2'],
                        help="Model architecture to use")
    
    args = parser.parse_args()
    
    if args.data_dir:
        DATA_DIR = args.data_dir
    
    if args.train:
        train_model(args.model)
    elif args.predict:
        predict(args.predict)
    else:
        # Simple interaction
        mode = input("Select Mode: (1) Train, (2) Predict: ")
        if mode == "1":
            d_dir = input(f"Enter data directory (default: {DATA_DIR}): ").strip()
            if d_dir:
                DATA_DIR = d_dir
            if d_dir:
                DATA_DIR = d_dir
            m_name = input("Enter model (resnet18, resnet50, resnet101, googlenet, inception_v3, inceptionresnetv2) [default: resnet50]: ").strip()
            if not m_name: m_name = "resnet50"
            train_model(m_name)
        elif mode == "2":
            f = input("Enter path to wav file: ").strip('"')
            predict(f)
