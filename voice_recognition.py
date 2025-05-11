import pandas as pd
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
import io
import requests
from pathlib import Path

class AudioTransform:
    def __init__(self, target_length=16000):
        self.target_length = target_length
        self.resample = T.Resample(orig_freq=16000, new_freq=16000)
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=16000,
            n_fft=1024,
            hop_length=512,
            n_mels=128,
            f_min=20,
            f_max=8000
        )
        self.amplitude_to_db = T.AmplitudeToDB()
    
    def __call__(self, waveform):
        # Apply transforms sequentially
        waveform = self.resample(waveform)
        
        # Ensure waveform has target length
        if waveform.shape[1] > self.target_length:
            # Random crop if longer
            start = torch.randint(0, waveform.shape[1] - self.target_length, (1,))
            waveform = waveform[:, start:start + self.target_length]
        elif waveform.shape[1] < self.target_length:
            # Pad with zeros if shorter
            padding = self.target_length - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, padding))
        
        # Convert to mel spectrogram
        mel_spec = self.mel_spectrogram(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)
        
        return mel_spec_db

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class EmotionRecognitionModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionRecognitionModel, self).__init__()
        self.num_classes = num_classes
        
        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        
        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        # Add channel dimension if needed
        if len(x.shape) == 3:  # [batch, time, features]
            x = x.unsqueeze(1)  # Add channel dimension
        
        # Initial convolution
        x = self.conv1(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        return x

class AudioEmotionDataset(Dataset):
    def __init__(self, df, transform=None, base_path="dataset"):
        self.df = df
        self.transform = transform
        self.base_path = base_path
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        # Get the audio path and emotion
        audio_path = self.df.iloc[idx]['path']
        emotion = self.df.iloc[idx]['emotion_encoded']
        
        try:
            # Construct full path
            full_path = os.path.join(self.base_path, audio_path)
            
            # Check if file exists
            if not os.path.exists(full_path):
                print(f"File not found: {full_path}")
                return torch.zeros((1, 128, 128)), emotion
            
            # Load audio file
            waveform, sample_rate = torchaudio.load(full_path)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Apply transformations if any
            if self.transform:
                waveform = self.transform(waveform)
                
            return waveform, emotion
            
        except Exception as e:
            print(f"Error loading file {full_path}: {str(e)}")
            # Return a zero tensor as fallback
            return torch.zeros((1, 128, 128)), emotion

def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1)
        correct += pred.eq(target).sum().item()
        total += target.size(0)
        
    return total_loss / len(train_loader), 100. * correct / total

def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    return total_loss / len(test_loader), 100. * correct / total

def download_and_save_audio_files(df, output_dir="dataset"):
    """Download and save audio files from the dataset"""
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save metadata
    metadata_path = output_dir / "metadata.parquet"
    if not metadata_path.exists():
        print(f"\nSaving metadata to {metadata_path}...")
        df.to_parquet(metadata_path)
    
    print(f"\nDownloading audio files to {output_dir}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # Get audio data and path
        audio_data = row['speech']['bytes']
        audio_path = row['path']
        
        # Create full path
        full_path = output_dir / audio_path
        
        # Skip if file already exists
        if full_path.exists():
            continue
        
        # Create parent directories if they don't exist
        full_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save audio file
        try:
            with open(full_path, 'wb') as f:
                f.write(audio_data)
        except Exception as e:
            print(f"Error saving file {full_path}: {str(e)}")
    
    print(f"\nAudio files saved to {output_dir}")
    return output_dir

def load_and_prepare_dataset():
    print("Loading dataset from parquet files...")
    
    # Define paths
    dataset_dir = Path("dataset")
    metadata_path = dataset_dir / "metadata.parquet"
    
    # Check if dataset is already downloaded
    if metadata_path.exists():
        print("Found existing dataset metadata, loading...")
        train_df = pd.read_parquet(metadata_path)
        test_df = pd.read_parquet(metadata_path)  # We'll split it later
        
        # Create label encoder
        label_encoder = LabelEncoder()
        train_df['emotion_encoded'] = label_encoder.fit_transform(train_df['emotion'])
        test_df['emotion_encoded'] = label_encoder.transform(test_df['emotion'])
        
        # Split into train and test
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df['emotion'])
        
    else:
        with tqdm(total=100, desc="Loading dataset") as pbar:
            # Define paths to parquet files
            splits = {
                'train': 'data/train-00000-of-00001-1f5fe73d1293189c.parquet',
                'test': 'data/test-00000-of-00001-a2b788d59856c4ae.parquet'
            }
            
            # Load train and test data
            train_df = pd.read_parquet("hf://datasets/Aniemore/resd_annotated/" + splits["train"])
            pbar.update(40)
            
            test_df = pd.read_parquet("hf://datasets/Aniemore/resd_annotated/" + splits["test"])
            pbar.update(30)
            
            # Create label encoder
            label_encoder = LabelEncoder()
            train_df['emotion_encoded'] = label_encoder.fit_transform(train_df['emotion'])
            test_df['emotion_encoded'] = label_encoder.transform(test_df['emotion'])
            pbar.update(30)
            
            # Download and save audio files
            download_and_save_audio_files(pd.concat([train_df, test_df]))
    
    # Print dataset information
    print("\nDataset loaded successfully")
    print(f"Training set size: {len(train_df)}")
    print(f"Test set size: {len(test_df)}")
    
    return train_df, test_df, label_encoder

def main(train_new_model=False):
    # Load and prepare dataset
    train_df, test_df, label_encoder = load_and_prepare_dataset()
    
    print("\nUnique emotions:", label_encoder.classes_)
    print("\nEmotion distribution in training set:")
    print(train_df['emotion'].value_counts())
    
    # Create transform instance with target length (1 second at 16kHz)
    audio_transform = AudioTransform(target_length=16000)
    
    # Create datasets with base path
    base_path = "dataset"  # Path to downloaded audio files
    train_dataset = AudioEmotionDataset(train_df, transform=audio_transform, base_path=base_path)
    test_dataset = AudioEmotionDataset(test_df, transform=audio_transform, base_path=base_path)
    
    # Create data loaders with num_workers=0 to avoid potential issues
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=0)
    
    # Initialize model
    num_classes = len(label_encoder.classes_)
    model = EmotionRecognitionModel(num_classes)
    
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Load pre-trained model if exists and not training new
    model_path = Path("models/emotion_recognition_model.pth")
    if not train_new_model and model_path.exists():
        print("\nLoading pre-trained model...")
        model.load_state_dict(torch.load(model_path))
        print("Model loaded successfully")
        
        # Evaluate loaded model
        criterion = nn.CrossEntropyLoss()
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        print(f'\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        return
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3)
    
    # Print model summary
    print("\nModel architecture:")
    print(model)
    
    # Training loop
    num_epochs = 50
    best_acc = 0
    patience = 10
    no_improve = 0
    
    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)
        
        # Update learning rate
        scheduler.step(test_acc)
        
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%')
        print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%')
        
        # Early stopping
        if test_acc > best_acc:
            best_acc = test_acc
            no_improve = 0
            torch.save(model.state_dict(), 'models/best_emotion_recognition_model.pth')
            print(f'New best model saved with accuracy: {best_acc:.2f}%')
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    # Save final model and label encoder
    torch.save(model.state_dict(), 'models/emotion_recognition_model.pth')
    import joblib
    joblib.dump(label_encoder, 'models/label_encoder.joblib')
    print("\nFinal model saved as 'models/emotion_recognition_model.pth'")
    print("LabelEncoder saved as 'models/label_encoder.joblib'")

if __name__ == "__main__":
    # Create models directory if it doesn't exist
    Path("models").mkdir(exist_ok=True)
    
    # By default, load pre-trained model
    main(train_new_model=False)
    
    # To train a new model, uncomment the following line:
    # main(train_new_model=True) 