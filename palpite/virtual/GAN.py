import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist
import time
from datetime import datetime
import os
import sqlite3

class IntegerArrayDataset(Dataset):
   def __init__(self, data_tensor):
       self.data = data_tensor
   
   def __len__(self):
       return len(self.data)
   
   def __getitem__(self, idx):
       return self.data[idx]

def load_and_split_data(csv_path, test_size=100):
   df = pd.read_csv(csv_path, header=None)
   test_df = df.iloc[-test_size:, 1:].values
   train_df = df.iloc[:-test_size, 1:].values
   
   train_tensor = torch.FloatTensor(train_df)
   test_tensor = torch.FloatTensor(test_df)
   
   train_tensor = (train_tensor - 1) / 24.0
   test_tensor = (test_tensor - 1) / 24.0
   
   return train_tensor, test_tensor

def load_gan_config(user_id=1):
   conn = sqlite3.connect('gan_config.db')
   cursor = conn.cursor()
   
   cursor.execute('''
   SELECT latent_dim, temperature, dropout_rate, learning_rate, 
          adam_beta1, adam_beta2, num_epochs, batch_size,
          test_size, num_samples, distance_metric,
          gen_layer1, gen_layer2, gen_layer3,
          disc_layer1, disc_layer2, disc_layer3,
          training_seed
   FROM gan_configs 
   WHERE user_id = ?
   ORDER BY modified_at DESC 
   LIMIT 1
   ''', (user_id,))
   
   result = cursor.fetchone()
   conn.close()
   
   if result is None:
       raise ValueError(f"No configuration found for user {user_id}")
       
   return {
       'latent_dim': result[0],
       'temperature': result[1], 
       'dropout_rate': result[2],
       'learning_rate': result[3],
       'adam_beta1': result[4],
       'adam_beta2': result[5],
       'num_epochs': result[6],
       'batch_size': result[7],
       'test_size': result[8],
       'num_samples': result[9],
       'distance_metric': result[10],
       'gen_layer1': result[11],
       'gen_layer2': result[12], 
       'gen_layer3': result[13],
       'disc_layer1': result[14],
       'disc_layer2': result[15],
       'disc_layer3': result[16],
       'training_seed': result[17]
   }

class Generator(nn.Module):
   def __init__(self, config):
       super(Generator, self).__init__()
       
       self.model = nn.Sequential(
           nn.Linear(config['latent_dim'], config['gen_layer1']),
           nn.BatchNorm1d(config['gen_layer1']),
           nn.LeakyReLU(0.2),
           
           nn.Linear(config['gen_layer1'], config['gen_layer2']),
           nn.BatchNorm1d(config['gen_layer2']),
           nn.LeakyReLU(0.2),
           
           nn.Linear(config['gen_layer2'], config['gen_layer3']),
           nn.BatchNorm1d(config['gen_layer3']),
           nn.LeakyReLU(0.2),
           
           nn.Linear(config['gen_layer3'], 25)
       )
       
       self.temperature = config['temperature']
       
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
   def __init__(self, config):
       super(Discriminator, self).__init__()
       
       self.model = nn.Sequential(
           nn.Linear(15, config['disc_layer1']),
           nn.LeakyReLU(0.2),
           nn.Dropout(config['dropout_rate']),
           
           nn.Linear(config['disc_layer1'], config['disc_layer2']),
           nn.LeakyReLU(0.2),
           nn.Dropout(config['dropout_rate']),
           
           nn.Linear(config['disc_layer2'], config['disc_layer3']),
           nn.LeakyReLU(0.2),
           nn.Dropout(config['dropout_rate']),
           
           nn.Linear(config['disc_layer3'], 1),
           nn.Sigmoid()
       )
   
   def forward(self, x):
       return self.model(x)

def denormalize_and_discretize(tensor):
   denorm = tensor * 24.0 + 1.0
   return torch.round(denorm).clamp(1, 25)

def train_gan(generator, discriminator, dataloader, config):
   criterion = nn.BCELoss()
   g_optimizer = optim.Adam(generator.parameters(), lr=config['learning_rate'], 
                          betas=(config['adam_beta1'], config['adam_beta2']))
   d_optimizer = optim.Adam(discriminator.parameters(), lr=config['learning_rate'], 
                          betas=(config['adam_beta1'], config['adam_beta2']))
   
   generator.train()
   discriminator.train()
   
   for epoch in range(config['num_epochs']):
       for i, real_arrays in enumerate(dataloader):
           batch_size = real_arrays.size(0)
           
           d_optimizer.zero_grad()
           
           real_labels = torch.ones(batch_size, 1)
           fake_labels = torch.zeros(batch_size, 1)
           
           outputs = discriminator(real_arrays)
           d_loss_real = criterion(outputs, real_labels)
           
           noise = torch.randn(batch_size, config['latent_dim'])
           fake_arrays = generator(noise)
           outputs = discriminator(fake_arrays.detach())
           d_loss_fake = criterion(outputs, fake_labels)
           
           d_loss = d_loss_real + d_loss_fake
           d_loss.backward()
           d_optimizer.step()
           
           g_optimizer.zero_grad()
           outputs = discriminator(fake_arrays)
           g_loss = criterion(outputs, real_labels)
           g_loss.backward()
           g_optimizer.step()
       
       if (epoch + 1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{config["num_epochs"]}]')

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

def calculate_hamming_distance(X, Y):
   return cdist(X, Y, metric='hamming')

def calculate_similarity_scores(generated_arrays, test_set, metric='euclidean'):
   generated_sorted = np.sort(generated_arrays, axis=1)
   test_sorted = np.sort(test_set, axis=1)
   
   if metric == 'euclidean':
       distances = euclidean_distances(generated_sorted, test_sorted)
   elif metric == 'manhattan':
       distances = manhattan_distances(generated_sorted, test_sorted)
   elif metric == 'hamming':
       distances = calculate_hamming_distance(generated_sorted, test_sorted)
   else:
       raise ValueError(f"Métrica {metric} não suportada")
   
   min_distances = np.min(distances, axis=1)
   
   if metric == 'hamming':
       similarity_scores = 1 - min_distances
   else:
       similarity_scores = 1 / (1 + min_distances)
   
   return similarity_scores

def generate_and_sort_samples(generator, test_tensor, config):
   generator.eval()
   
   with torch.no_grad():
       noise = torch.randn(config['num_samples'], config['latent_dim'])
       generated = generator(noise)
       final_arrays = denormalize_and_discretize(generated)
       samples = final_arrays.numpy().astype(int)
       
       if not verify_unique_numbers(samples):
           print("Erro: Amostras geradas contêm números duplicados ou inválidos")
           return None
       
       test_arrays = denormalize_and_discretize(test_tensor).numpy()
       similarity_scores = calculate_similarity_scores(samples, test_arrays, config['distance_metric'])
       sorted_indices = np.argsort(similarity_scores)[::-1]
       sorted_samples = samples[sorted_indices]
       
       return sorted_samples, similarity_scores[sorted_indices]

def save_to_csv(arrays, scores, filename='generated_arrays.csv'):
   with open(filename, 'w', newline='') as f:
       for arr, score in zip(arrays, scores):
           sorted_arr = sorted(arr)
           line = ','.join(map(str, sorted_arr))
           f.write(f"{line}\n")

if __name__ == "__main__":
   try:
       # Load configuration
       config = load_gan_config()
       
       # Clean up previous output
       if os.path.exists('gan_generated_lotofacil.csv'):
           os.remove('gan_generated_lotofacil.csv')
       
       # Set random seeds
       print("Setting seeds...")
       torch.manual_seed(config['training_seed'])
       np.random.seed(config['training_seed'])
       
       # Load and prepare data
       print("Loading and splitting data...")
       train_tensor, test_tensor = load_and_split_data('lotofacil.csv', test_size=config['test_size'])
       
       # Prepare dataset and dataloader
       dataset = IntegerArrayDataset(train_tensor)
       batch_size = min(len(dataset), config['batch_size'])
       dataloader = DataLoader(
           dataset, 
           batch_size=batch_size,
           shuffle=True,
           drop_last=True
       )
       
       # Initialize models
       generator = Generator(config)
       discriminator = Discriminator(config)
       
       # Train models
       print(f"Starting training with {len(train_tensor)} samples...")
       print(f"Test set size: {config['test_size']}")
       print(f"Distance metric: {config['distance_metric']}")
       
       start_time = time.time()
       train_gan(generator, discriminator, dataloader, config)
       train_duration = time.time() - start_time
       print(f"Training completed in {train_duration:.2f} seconds")
       
       # Generate samples
       print("\nGenerating and sorting samples...")
       start_time = time.time()
       result = generate_and_sort_samples(generator, test_tensor, config)
       
       if result is not None:
           sorted_samples, similarity_scores = result
           save_to_csv(sorted_samples, similarity_scores)
           
           gen_duration = time.time() - start_time
           print(f"Generation completed in {gen_duration:.2f} seconds")
           
           # Display sample results
           print("\nTop 5 samples by similarity:")
           for i in range(min(5, len(sorted_samples))):
               print(f"Sample {i+1}: {sorted(sorted_samples[i])} (Score: {similarity_scores[i]:.4f})")
       else:
           print("Error: Sample generation failed")
           
   except Exception as e:
       print(f"Error: {e}")
       raise