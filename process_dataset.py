import os
import torch
import librosa
import torchaudio
import soundfile as sf

# Directories
original_dir = 'data/original_recordings'
split_dir = 'data/original'
subsampled_dir = 'data/subsampled'

sample_length = 512
step = 512

os.makedirs(split_dir, exist_ok=True)
os.makedirs(subsampled_dir, exist_ok=True)

def split_and_save_audio(input_path, output_dir, sample_length, step):
    # Load the audio file
    data, sr = librosa.load(input_path, sr=96000, mono=False)
    
    total_samples = data.shape[1]
    num_splits = (total_samples - sample_length) // step + 1
    
    for i in range(num_splits):
        start_sample = i * step
        end_sample = start_sample + sample_length
        split_data = data[:, start_sample:end_sample]
        
        # Construct the output filename
        base_filename = os.path.basename(input_path)
        split_filename = f"{os.path.splitext(base_filename)[0]}_split_{i}.wav"
        split_path = os.path.join(output_dir, split_filename)
        
        # Save the split audio
        sf.write(split_path, split_data.T, sr)

def subsample_audio(input_path, output_path):
    # Load the audio file
    data, sr = librosa.load(input_path, sr=96000, mono=False)
    
    # Subsample: Keep every other channel
    subsampled_data = data[::2]
    
    # Save the subsampled audio
    sf.write(output_path, subsampled_data.T, sr)

def compute_and_save_fft(directory, n_fft=2048):
    for filename in os.listdir(directory):
        if filename.endswith('.wav'):
            file_path = os.path.join(directory, filename)
            waveform, _ = torchaudio.load(file_path, backend="soundfile")
            
            fft_result = torch.fft.fft(waveform, n=n_fft)
            magnitude = torch.abs(fft_result)
            phase = torch.angle(fft_result)
            phase_cos = torch.cos(phase)
            phase_sin = torch.sin(phase)
            fft_features = torch.stack([magnitude, phase_cos, phase_sin], dim=0)
            
            # Save FFT features
            torch.save(fft_features, file_path.replace('.wav', '.pt'))

# Process each WAV file in the original directory for splitting
for filename in os.listdir(original_dir):
    if filename.endswith('.wav'):
        input_path = os.path.join(original_dir, filename)
        split_and_save_audio(input_path, split_dir, sample_length, step)

# Process each split WAV file for subsampling
for filename in os.listdir(split_dir):
    if filename.endswith('.wav'):
        input_path = os.path.join(split_dir, filename)
        output_path = os.path.join(subsampled_dir, filename.replace("_split", "_subsampled"))
        
        # Subsample and save the audio
        subsample_audio(input_path, output_path)
        print(f"Processed and subsampled {filename}")

compute_and_save_fft(split_dir)
compute_and_save_fft(subsampled_dir)
