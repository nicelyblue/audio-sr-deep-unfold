import librosa
import torchaudio


# Path to your audio file
audio_file_path = 'data/subsampled/scenario_1.wav'

audio, _ = torchaudio.load(audio_file_path, backend="soundfile")

# Load the audio file
# audio, sr = librosa.load(audio_file_path, sr=None, mono=False)

# Output the shape
# For mono audio, `audio` will be a 1D array (samples,)
# For stereo or more channels, `audio` will be a 2D array (channels, samples)
print(f"Shape: {audio.shape}")

