import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio

class AudioDataset(Dataset):
    def __init__(self, input_paths, target_paths, transform=None):
        self.input_paths = input_paths
        self.target_paths = target_paths
        self.transform = transform

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        input_path = self.input_paths[idx]
        target_path = self.target_paths[idx]
        
        input_waveform, _ = torchaudio.load(input_path)
        target_waveform, _ = torchaudio.load(target_path)

        if self.transform:
            input_waveform = self.transform(input_waveform[0])  # Assuming mono audio
            target_waveform = self.transform(target_waveform[0]) # TODO: What if not mono?

        return input_waveform, target_waveform


def fft_transform(waveform, n_fft=2048):
    fft_result = torch.fft.fft(waveform, n=n_fft)
    magnitude = torch.abs(fft_result)
    phase = torch.angle(fft_result)

    phase_cos = torch.cos(phase)
    phase_sin = torch.sin(phase)

    return torch.stack([magnitude, phase_cos, phase_sin], dim=0)