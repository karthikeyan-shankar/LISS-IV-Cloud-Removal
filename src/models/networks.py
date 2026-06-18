import torch
import torch.nn as nn

class UNetBlock(nn.Module):
    """
    Sub-block of the U-Net architecture containing Convolution, BatchNorm, and Activation.
    """
    def __init__(self, in_channels, out_channels, down=True, act="relu", use_dropout=False):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False, padding_mode="reflect")
            if down else
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU() if act == "relu" else nn.LeakyReLU(0.2)
        )
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        return self.dropout(x) if self.use_dropout else x


class Generator(nn.Module):
    """
    U-Net Generator for Image Translation.
    Inputs: 6 channels (3-band Cloudy Today + 3-band Clear History).
    Outputs: 3 channels (3-band Reconstructed Today).
    """
    def __init__(self, in_channels=6, out_channels=3, features=64):
        super().__init__()
        # Encoder (Downsampling)
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        self.down1 = UNetBlock(features, features * 2, down=True, act="leaky")
        self.down2 = UNetBlock(features * 2, features * 4, down=True, act="leaky")
        self.down3 = UNetBlock(features * 4, features * 8, down=True, act="leaky")
        self.down4 = UNetBlock(features * 8, features * 8, down=True, act="leaky")
        self.down5 = UNetBlock(features * 8, features * 8, down=True, act="leaky")
        self.down6 = UNetBlock(features * 8, features * 8, down=True, act="leaky")
        
        # Bottleneck (deepest layer)
        self.bottleneck = nn.Sequential(
            nn.Conv2d(features * 8, features * 8, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder (Upsampling with skip connections)
        self.up1 = UNetBlock(features * 8, features * 8, down=False, act="relu", use_dropout=True)
        self.up2 = UNetBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up3 = UNetBlock(features * 16, features * 8, down=False, act="relu", use_dropout=True)
        self.up4 = UNetBlock(features * 16, features * 8, down=False, act="relu")
        self.up5 = UNetBlock(features * 16, features * 4, down=False, act="relu")
        self.up6 = UNetBlock(features * 8, features * 2, down=False, act="relu")
        self.up7 = UNetBlock(features * 4, features, down=False, act="relu")
        
        # Final output layer
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(features * 2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Scales output pixels between -1.0 and 1.0 (or we use Sigmoid/Clipping if 0-1)
        )

    def forward(self, x):
        # Encoder
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        
        # Bottleneck
        bn = self.bottleneck(d7)
        
        # Decoder with Skip Connections (concatenation along channel dimension)
        u1 = self.up1(bn)
        u2 = self.up2(torch.cat([u1, d7], dim=1))
        u3 = self.up3(torch.cat([u2, d6], dim=1))
        u4 = self.up4(torch.cat([u3, d5], dim=1))
        u5 = self.up5(torch.cat([u4, d4], dim=1))
        u6 = self.up6(torch.cat([u5, d3], dim=1))
        u7 = self.up7(torch.cat([u6, d2], dim=1))
        
        # Output
        out = self.final_up(torch.cat([u7, d1], dim=1))
        return out


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator.
    Classifies localized image patches of size 70x70 as real or fake.
    Inputs: 9 channels (6 channels of input [Cloudy + History] + 3 channels of target/generated).
    """
    def __init__(self, in_channels=9, features=[64, 128, 256, 512]):
        super().__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, features[0], kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2)
        )
        
        layers = []
        in_c = features[0]
        for feature in features[1:]:
            layers.append(
                nn.Sequential(
                    nn.Conv2d(in_c, feature, kernel_size=4, stride=1 if feature == features[-1] else 2, padding=1, bias=False, padding_mode="reflect"),
                    nn.BatchNorm2d(feature),
                    nn.LeakyReLU(0.2)
                )
            )
            in_c = feature
            
        self.model = nn.Sequential(*layers)
        
        # Single channel patch output grid
        self.final = nn.Conv2d(in_c, 1, kernel_size=4, stride=1, padding=1, padding_mode="reflect")

    def forward(self, x, y):
        # Concatenate inputs and target along channel dimension
        xy = torch.cat([x, y], dim=1)
        x = self.initial(xy)
        x = self.model(x)
        out = self.final(x)
        return out

if __name__ == "__main__":
    # Test architectures with mock tensors
    x = torch.randn((1, 6, 256, 256))
    gen = Generator()
    pred = gen(x)
    print("Generator Output Shape:", pred.shape) # Expected: (1, 3, 256, 256)
    
    y = torch.randn((1, 3, 256, 256))
    disc = Discriminator()
    disc_pred = disc(x, y)
    print("Discriminator Output Shape:", disc_pred.shape) # Expected Patch Grid: (1, 1, 30, 30)
