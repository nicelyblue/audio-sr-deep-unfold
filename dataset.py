import torch
from torch.utils.data import Dataset

class AudioDataset(Dataset):
    def __init__(self, input_paths, target_paths):
        self.input_paths = input_paths
        self.target_paths = target_paths

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_fft_path = self.input_paths[idx]
        target_fft_path = self.target_paths[idx]
        
        # Load pre-computed FFT features
        input_fft = torch.load(input_fft_path)
        target_fft = torch.load(target_fft_path)

        return input_fft, target_fft
