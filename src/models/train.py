import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import LISS4Dataset
from networks import Generator, Discriminator
import numpy as np

# --- CONFIGURATION ---
BATCH_SIZE = 8        # Set to 8 or 16 for RTX 2050
LEARNING_RATE = 2e-4
L1_LAMBDA = 100       # Weight for spatial L1 pixel reconstruction loss
NDVI_LAMBDA = 50      # Weight for our unique NDVI spectral consistency loss
NUM_EPOCHS = 30       # 30 epochs is plenty for 3500 patches
SAVE_INTERVAL = 5     # Save weights every 5 epochs

# Custom NDVI Loss for Scientific Spectral Accuracy
class NDVILoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
        self.l1 = nn.L1Loss()
        
    def calculate_ndvi(self, img):
        # img shape: (batch, bands, height, width)
        # Scale pixels from Tanh [-1, 1] to [0, 1]
        img_norm = (img + 1.0) / 2.0
        
        red = img_norm[:, 1, :, :]
        nir = img_norm[:, 2, :, :]
        
        ndvi = (nir - red) / (nir + red + self.epsilon)
        return ndvi

    def forward(self, generated, target):
        ndvi_gen = self.calculate_ndvi(generated)
        ndvi_target = self.calculate_ndvi(target)
        return self.l1(ndvi_gen, ndvi_target)

def train():
    # Force PyTorch to use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training LISS-IV Cloud Removal GAN on: {device}")
    if torch.cuda.is_available():
        print(f"  GPU Device Name: {torch.cuda.get_device_name(0)}")
    
    # Initialize U-Net Generator & PatchGAN Discriminator
    gen = Generator(in_channels=6, out_channels=3).to(device)
    disc = Discriminator(in_channels=9).to(device)
    
    # Load latest checkpoint if available to resume training
    start_epoch = 1
    import glob
    checkpoints = glob.glob("models/generator_epoch_*.pth")
    if checkpoints:
        try:
            checkpoints.sort(key=lambda x: int(os.path.basename(x).replace("generator_epoch_", "").replace(".pth", "")))
            latest_checkpoint = checkpoints[-1]
            start_epoch = int(os.path.basename(latest_checkpoint).replace("generator_epoch_", "").replace(".pth", "")) + 1
            gen.load_state_dict(torch.load(latest_checkpoint, map_location=device))
            print(f"RESUMING TRAINING: Loaded generator checkpoint: {latest_checkpoint}, resuming from Epoch {start_epoch}")
        except Exception as e:
            print(f"Could not load checkpoint to resume: {e}. Starting from scratch.")
    
    # Optimizers
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE, betas=(0.5, 0.999))
    
    # Loss functions
    BCE = nn.BCEWithLogitsLoss()
    L1 = nn.L1Loss()
    NDVI = NDVILoss()
    
    # Prepare Dataloader using the 3500+ clear patches
    clear_path = os.path.join("data", "patches", "clear")
    
    if not os.path.exists(clear_path) or len(os.listdir(clear_path)) == 0:
        raise FileNotFoundError(f"No patched LISS-IV images found in {clear_path}. Please run run_patching.py first.")

    dataset = LISS4Dataset(clear_dir=clear_path)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True if torch.cuda.is_available() else False, num_workers=2)
    
    os.makedirs("models", exist_ok=True)
    
    print(f"Starting training loop. Total patches to train: {len(dataset)}")
    
    # Training Loop
    for epoch in range(start_epoch, NUM_EPOCHS + 1):
        for idx, batch in enumerate(loader):
            # Load clear target, history, and cloudy input
            clear_img = batch['clear'].to(device)     # Ground truth
            history_img = batch['history'].to(device) # Reference
            cloudy_img = batch['cloudy'].to(device)   # Input with simulated clouds
            
            # Scale pixels from [0, 1] to [-1, 1] to match GAN Generator Tanh output
            clear_img = clear_img * 2.0 - 1.0
            history_img = history_img * 2.0 - 1.0
            cloudy_img = cloudy_img * 2.0 - 1.0
            
            # Combine Input 1 (Cloudy) and Input 2 (History Reference) along channel dimension
            gen_input = torch.cat([cloudy_img, history_img], dim=1) # Shape: (B, 6, H, W)
            
            # ---------------------------
            #  1. Train Discriminator
            # ---------------------------
            # Generate fake image
            fake_img = gen(gen_input)
            
            # Prediction on Real pair
            disc_real = disc(gen_input, clear_img)
            loss_disc_real = BCE(disc_real, torch.ones_like(disc_real) * 0.9) # Label smoothing (0.9 instead of 1.0)
            
            # Prediction on Fake pair
            disc_fake = disc(gen_input, fake_img.detach())
            loss_disc_fake = BCE(disc_fake, torch.zeros_like(disc_fake))
            
            loss_disc = (loss_disc_real + loss_disc_fake) / 2
            
            opt_disc.zero_grad()
            loss_disc.backward()
            opt_disc.step()
            
            # ---------------------------
            #  2. Train Generator
            # ---------------------------
            # GAN adversarial loss
            disc_fake = disc(gen_input, fake_img)
            loss_gen_adv = BCE(disc_fake, torch.ones_like(disc_fake))
            
            # Spatial reconstruction loss (L1)
            loss_gen_l1 = L1(fake_img, clear_img) * L1_LAMBDA
            
            # Scientific Spectral NDVI loss
            loss_gen_ndvi = NDVI(fake_img, clear_img) * NDVI_LAMBDA
            
            # Total Loss
            loss_gen = loss_gen_adv + loss_gen_l1 + loss_gen_ndvi
            
            opt_gen.zero_grad()
            loss_gen.backward()
            opt_gen.step()
            
            if idx % 50 == 0:
                print(f"Epoch [{epoch}/{NUM_EPOCHS}] Batch {idx}/{len(loader)} | Loss D: {loss_disc.item():.4f} | Loss G: {loss_gen.item():.4f} (L1: {loss_gen_l1.item():.2f}, NDVI: {loss_gen_ndvi.item():.2f})")
                
        # Save checkpoints
        if epoch % SAVE_INTERVAL == 0:
            torch.save(gen.state_dict(), f"models/generator_epoch_{epoch}.pth")
            print(f"Saved Generator checkpoint: models/generator_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()
