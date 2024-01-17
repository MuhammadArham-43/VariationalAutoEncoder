import torch
import torch.nn as nn
from torch.optim import Adam
import torchvision.transforms as transforms
from torchvision.datasets import MNIST
from torch.utils.data import Dataset, DataLoader
from model import VAE
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from utils import save_image_grid, load_config


if __name__ == "__main__":
    CONFIG_PATH = "configs/mnist_model_config.yaml"
    model_config = load_config(CONFIG_PATH)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    INPUT_DIM = model_config.model.input_dim
    HIDDEN_DIM = model_config.model.hidden_dim
    Z_DIM = model_config.model.z_dim
    NUM_EPOCHS = model_config.training.num_epochs
    BATCH_SIZE = model_config.training.batch_size
    LR = model_config.training.learning_rate
    
    MODEL_SAVE_DIR = model_config.save_directories.model_save_dir
    IMG_SAVE_DIR = model_config.save_directories.imgs_save_dir
    
    rows = model_config.visualization.grid_rows
    columns = model_config.visualization.grid_cols
    
    
    assert (rows * columns) == BATCH_SIZE
    
    
    train_dataset = MNIST(root=".data", train=True, transform=transforms.ToTensor(), download=True)
    val_dataset = MNIST(root=".data", train=False, transform=transforms.ToTensor(), download=True)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    net = VAE(
        input_size=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        z_dim=Z_DIM
    ).to(DEVICE)
    
    optimizer = Adam(net.parameters(), lr=LR)
    
    criterion = nn.BCELoss(reduction="sum")
    
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    os.makedirs(IMG_SAVE_DIR, exist_ok=True)
    for epoch in range(NUM_EPOCHS):
        net.eval()
        imgs, _ = next(iter(val_dataloader))
        with torch.no_grad():
            imgs = imgs.to(DEVICE).view(-1, INPUT_DIM)
            out, mu, std = net(imgs)
            out_reshaped = out.reshape((-1, 28, 28))
            
            save_path = os.path.join(IMG_SAVE_DIR, f"res_epoch{epoch}.png")
            save_image_grid(out_reshaped, save_path, rows, columns)
            
            
        
        running_loss = 0
        net.train()
        for idx, batch in enumerate(tqdm(train_dataloader)): 
            imgs, _ = batch
            imgs = imgs.to(DEVICE).view(-1, INPUT_DIM)
            
            out, mu, std = net(imgs)
            
            reconstruction_loss = criterion(out, imgs)
            kl_divergence_loss = - torch.sum(1 + torch.log(std.pow(2)) - mu.pow(2) - std.pow(2))
            loss = reconstruction_loss + kl_divergence_loss
            running_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f"EPOCH {epoch + 1} || Loss: {running_loss / len(train_dataloader)}")
        if (epoch + 1) % 10 == 0:
            save_path = os.path.join(MODEL_SAVE_DIR, f"vae_epoch{epoch + 1}.pth")
            torch.save(net.state_dict(), save_path)
    
            