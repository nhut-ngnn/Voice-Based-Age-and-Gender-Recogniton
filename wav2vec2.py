import os
import pandas as pd
from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
import torchaudio

# Initialize the Wav2Vec2 processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")

# Function to extract features from an audio file
def extract_features(file_path):
    waveform, sample_rate = torchaudio.load(file_path)

    # If the sample rate is not 16000, resample the audio
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Ensure the audio is in mono by averaging across channels
    waveform = waveform.mean(dim=0, keepdim=True).squeeze()
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

    with torch.no_grad():
        features = model(**inputs).last_hidden_state

    features_np = features.squeeze().cpu().numpy()
    return features_np

# Folder containing audio files
audio_folder = "C:/Users/admin/Documents/AgeDetection/voice-bases-age-gender-classification/DataSet/ja/clips"
audio_files = [f for f in os.listdir(audio_folder) if f.endswith(".mp3")]

# List to store features and file names
data = []

# Loop through each audio file in the folder and extract features
for audio_file in audio_files:
    file_path = os.path.join(audio_folder, audio_file)
    features = extract_features(file_path)
    data.append({"file": audio_file, "features": features})

# Convert the list of features to a DataFrame
df = pd.DataFrame(data)

df.to_csv("audio_features.csv", index=False)
print("Features extracted and saved to audio_features.csv")
