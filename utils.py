import torch
import matplotlib.pyplot as plt
import yaml
from easydict import EasyDict

def save_image_grid(
    images: torch.Tensor,    # (batch, height, width)
    save_path: str,
    num_rows:int, 
    num_cols:int
):
    
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 6))
    for i in range(num_rows):
        for j in range(num_cols):
            index = i * num_cols + j
            axes[i, j].imshow(images[index], cmap='gray')
            axes[i, j].axis('off')

    # Save the entire grid as an image
    plt.savefig(save_path)
    plt.close()
    
    
def load_config(config_file_path):
    with open(config_file_path, 'r') as file:
        config = yaml.safe_load(file)
    return EasyDict(config)