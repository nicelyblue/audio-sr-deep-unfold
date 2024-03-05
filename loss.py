import torch.nn as nn

class MagnitudePhaseLoss(nn.Module):
    def __init__(self):
        super(MagnitudePhaseLoss, self).__init__()
        self.mse_loss = nn.MSELoss()

    def forward(self, predicted, target):
        magnitude_loss = self.mse_loss(predicted[:, 0, :, :], target[:, 0, :, :])
        phase_cos_loss = self.mse_loss(predicted[:, 1, :, :], target[:, 1, :, :])
        phase_sin_loss = self.mse_loss(predicted[:, 2, :, :], target[:, 2, :, :])

        total_loss = magnitude_loss + phase_cos_loss + phase_sin_loss
        return total_loss