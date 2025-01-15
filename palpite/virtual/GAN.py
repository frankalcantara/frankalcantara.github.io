import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances
import time
import os

class IntegerArrayDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def load_and_split_data(csv_path, test_size=100):
    # Read CSV file
    df = pd.read_csv(csv_path, header=None)
    
    # Split into train and test
    test_df = df.iloc[-test_size:, 1:].values
    train_df = df.iloc[:-test_size, 1:].values
    
    # Convert to tensor and normalize
    train_tensor = torch.FloatTensor(train_df)
    test_tensor = torch.FloatTensor(test_df)
    
    # Normalize
    train_tensor = (train_tensor - 1) / 24.0
    test_tensor = (test_tensor - 1) / 24.0
    
    return train_tensor, test_tensor

class Generator(nn.Module):
    def __init__(self, latent_dim=50):  # Reduced latent dimension
        super(Generator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),  # Reduced layer sizes
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            
            nn.Linear(128, 25)
        )
        
        self.temperature = 0.5
        
    def forward(self, z):
        batch_size = z.size(0)
        logits = self.model(z)
        
        selected = torch.zeros((batch_size, 15))
        available_mask = torch.ones_like(logits, dtype=torch.bool)
        
        for i in range(15):
            masked_logits = logits.clone()
            masked_logits[~available_mask] = -1e9
            
            noise = -torch.log(-torch.log(torch.rand_like(masked_logits)))
            gumbel_logits = (masked_logits + noise) / self.temperature
            probs = torch.softmax(gumbel_logits, dim=-1)
            
            selected_idx = torch.argmax(probs, dim=-1)
            
            selected[:, i] = selected_idx + 1
            available_mask[torch.arange(batch_size), selected_idx] = False
        
        return (selected - 1) / 24.0

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(15, 128),  # Reduced layer sizes
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

def denormalize_and_discretize(tensor):
    denorm = tensor * 24.0 + 1.0
    return torch.round(denorm).clamp(1, 25)

def train_gan(generator, discriminator, dataloader, num_epochs=150, latent_dim=50):  # Reduced epochs
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    generator.train()
    discriminator.train()
    
    for epoch in range(num_epochs):
        for i, real_arrays in enumerate(dataloader):
            batch_size = real_arrays.size(0)
            
            # Train Discriminator
            d_optimizer.zero_grad()
            
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            
            outputs = discriminator(real_arrays)
            d_loss_real = criterion(outputs, real_labels)
            
            noise = torch.randn(batch_size, latent_dim)
            fake_arrays = generator(noise)
            outputs = discriminator(fake_arrays.detach())
            d_loss_fake = criterion(outputs, fake_labels)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            d_optimizer.step()
            
            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_arrays)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}]')

def verify_unique_numbers(arrays):
    for i, arr in enumerate(arrays):
        unique_nums = set(arr)
        if len(unique_nums) != 15:
            print(f"Erro: Array {i} tem números duplicados: {arr}")
            return False
        if not all(1 <= x <= 25 for x in arr):
            print(f"Erro: Array {i} tem números fora do intervalo [1,25]: {arr}")
            return False
    return True

def calculate_similarity_scores(generated_arrays, test_set):
    generated_sorted = np.sort(generated_arrays, axis=1)
    test_sorted = np.sort(test_set, axis=1)
    
    distances = euclidean_distances(generated_sorted, test_sorted)
    min_distances = np.min(distances, axis=1)
    similarity_scores = 1 / (1 + min_distances)
    
    return similarity_scores

def generate_and_sort_samples(generator, test_tensor, num_samples=1000, latent_dim=50):
    generator.eval()
    
    # Reset random seeds for generation
    generation_seed = int(time.time())
    torch.manual_seed(generation_seed)
    np.random.seed(generation_seed)
    
    with torch.no_grad():
        # Generate samples
        noise = torch.randn(num_samples, latent_dim)
        generated = generator(noise)
        final_arrays = denormalize_and_discretize(generated)
        samples = final_arrays.numpy().astype(int)
        
        if not verify_unique_numbers(samples):
            print("Erro: Amostras geradas contêm números duplicados ou inválidos")
            return None
        
        test_arrays = denormalize_and_discretize(test_tensor).numpy()
        similarity_scores = calculate_similarity_scores(samples, test_arrays)
        sorted_indices = np.argsort(similarity_scores)[::-1]
        sorted_samples = samples[sorted_indices]
        
        return sorted_samples

def save_to_csv(arrays, filename='generated_arrays.csv'):
    with open(filename, 'w', newline='') as f:
        for arr in arrays:
            sorted_arr = sorted(arr)
            line = ','.join(map(str, sorted_arr))
            f.write(f"{line}\n")

if __name__ == "__main__":
    # Hyperparameters otimizados para CPU
    latent_dim = 100        # Reduzido de 100 para 50
    batch_size = 32        # Reduzido de 32 para 16
    num_epochs = 150       # Reduzido de 300 para 150
    test_size = 100

    os.remove('gan_generated_lotofacil.csv') if os.path.exists('gan_generated_lotofacil.csv') else None
    
    # Set random seeds only for training
    print("Definindo seeds para treinamento...")
    training_seed = generation_seed = int(time.time_ns()) % (2**32)
    torch.manual_seed(training_seed)
    np.random.seed(training_seed)
    
    # Load and split data
    print("Carregando e dividindo os dados...")
    train_tensor, test_tensor = load_and_split_data('lotofacil.csv', test_size=test_size)
    
    # Create dataset and dataloader
    dataset = IntegerArrayDataset(train_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize models
    generator = Generator(latent_dim)
    discriminator = Discriminator()
    
    # Train
    print(f"Iniciando treinamento com {len(train_tensor)} amostras...")
    print(f"Conjunto de teste: {test_size} amostras")
    start_time = time.time()
    
    train_gan(generator, discriminator, dataloader, num_epochs, latent_dim)
    
    end_time = time.time()
    print(f"Treinamento concluído em {end_time - start_time:.2f} segundos")
    
    # Generate and sort samples
    print("\nGerando e ordenando 50000 amostras...")
    start_time = time.time()
    
    sorted_samples = generate_and_sort_samples(
        generator, 
        test_tensor, 
        num_samples=50000, 
        latent_dim=latent_dim
    )
    
    if sorted_samples is not None:
        save_to_csv(sorted_samples, 'gan_generated_lotofacil.csv')
        
        end_time = time.time()
        print(f"Geração e ordenação concluídas em {end_time - start_time:.2f} segundos")
        
        print("\nAmostras geradas e ordenadas por similaridade (primeiras 5):")
        for i in range(min(10, len(sorted_samples))):
            print(f"Amostra {i+1}: {sorted(sorted_samples[i])}")
    else:
        print("Erro na geração das amostras.")