import torch 
import torch.nn as nn


class VAE(nn.Module):
    def __init__(
        self, 
        input_size:int, 
        hidden_dim: int = 256,
        z_dim:int = 32
    ) -> None:
        super().__init__()

        self.input_dim = input_size        
        self.latent_dim = z_dim
        
        self.encoder = nn.Linear(input_size, hidden_dim)
        self.mu_z = nn.Linear(hidden_dim, z_dim)
        
        self.std_z = nn.Linear(hidden_dim, z_dim)
        
        self.latent_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.decoder = nn.Linear(hidden_dim, input_size)
        
        self.activation = nn.ReLU()
    
    def encode(self, x):
        x = self.activation(self.encoder(x))
        mu, std = self.mu_z(x), self.std_z(x)
        return mu, std
    
    def decode(self, x):
        x = self.activation(self.latent_to_hidden(x))
        return torch.sigmoid(self.decoder(x))
    
    def forward(self, x):
        mu, std = self.encode(x)
        epsilon = torch.randn_like(std)
        latent = mu + (std * epsilon)
        return self.decode(latent), mu, std


if __name__ == "__main__":
    inp = torch.randn(16, 28 * 28)
    net = VAE(input_size=28*28)
    out, mu, std = net(inp)
    print(out.shape, mu.shape, std.shape)