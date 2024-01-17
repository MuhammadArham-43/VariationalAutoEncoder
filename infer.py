import torch
from PIL import Image
from model import VAE
from utils import save_image_grid, load_config

def generate_random_data(vae_model, num_samples):
    vae_model.eval()

    with torch.no_grad():
        latent_samples = torch.randn(num_samples, vae_model.latent_dim)

        generated_data = vae_model.decode(latent_samples)

    return generated_data



if __name__ == "__main__":
    CONFIG_PATH = "configs/mnist_model_config.yaml"
    model_config = load_config(CONFIG_PATH)
    
    NUM_SAMPLES = 128  # Adjust as needed
    NUM_ROWS = 8
    NUM_COLS = 16
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "runs/models/vae_epoch100.pth"
    
    net = VAE(
        input_size=model_config.model.input_dim,
        hidden_dim=model_config.model.hidden_dim,
        z_dim=model_config.model.z_dim
    )
    
    if MODEL_PATH:
        net.load_state_dict(torch.load(MODEL_PATH))
    
    generated_data = generate_random_data(net, NUM_SAMPLES)
    generated_data = generated_data.view((-1, 28, 28))
    save_image_grid(generated_data, "results.png", NUM_ROWS, NUM_COLS)
    