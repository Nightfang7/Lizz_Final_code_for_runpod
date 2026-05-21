import torch
import torch.nn as nn
import torchvision.models as models
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


class Autoencoder_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 16)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 16),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 3, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class SiameseNetwork_only_autoencoder(nn.Module):
    def __init__(self, encoder_path=None):
        super(SiameseNetwork_only_autoencoder, self).__init__()
        if encoder_path is None:
            raise ValueError("encoder_path must be provided for SiameseNetwork_only_autoencoder")
        self.autoencoder = Autoencoder_Conv().cuda()
        self.autoencoder.load_state_dict(torch.load(encoder_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        for param in self.autoencoder.parameters():
            param.requires_grad = False

    def forward(self, input1, input2):
        output1 = torch.squeeze(self.autoencoder.encoder(input1))
        output2 = torch.squeeze(self.autoencoder.encoder(input2))
        return output1, output2


class SiameseNetwork_autoencoder_Based(nn.Module):
    def __init__(self, encoder_path=None):
        super(SiameseNetwork_autoencoder_Based, self).__init__()
        if encoder_path is None:
            raise ValueError("encoder_path must be provided for SiameseNetwork_autoencoder_Based")
        self.autoencoder = Autoencoder_Conv().cuda()
        self.autoencoder.load_state_dict(torch.load(encoder_path, map_location='cuda' if torch.cuda.is_available() else 'cpu'))
        for param in self.autoencoder.parameters():
            param.requires_grad = False
        self.added_layers = nn.Sequential(
            nn.Linear(64, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 200),
            nn.ReLU(inplace=True),
            nn.Linear(200, 150))

    def forward(self, input1, input2):
        output1 = self.added_layers(torch.squeeze(self.autoencoder.encoder(input1)))
        output2 = self.added_layers(torch.squeeze(self.autoencoder.encoder(input2)))
        return output1, output2
