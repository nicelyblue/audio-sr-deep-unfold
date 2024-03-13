import os
import librosa
import soundfile as sf
import numpy as np

# Directories
original_dir = 'data/original'
subsampled_dir = 'data/subsampled'

# Create the subsampled directory if it doesn't exist
os.makedirs(subsampled_dir, exist_ok=True)

# Subsampling function
def subsample_audio(input_path, output_path):
    # Load the audio file
    data, sr = librosa.load(input_path, sr=None, mono=False)
    
    # Assuming data.shape is (channels, samples)
    # Subsample: Keep every other channel
    subsampled_data = data[::2]
    
    # Save the subsampled audio
    sf.write(output_path, subsampled_data.T, sr)

# Process each WAV file in the original directory
for filename in os.listdir(original_dir):
    if filename.endswith('.wav'):
        input_path = os.path.join(original_dir, filename)
        output_path = os.path.join(subsampled_dir, filename)
        
        # Subsample and save the audio
        subsample_audio(input_path, output_path)
        print(f"Processed {filename}")

