#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sqlite3
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances
from scipy.spatial.distance import cdist
import time
import logging
from datetime import datetime
import os
from typing import Tuple, List, Dict, Any

class IntegerArrayDataset(Dataset):
    def __init__(self, data_tensor):
        self.data = data_tensor
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

def setup_logging() -> logging.Logger:
    """Configure logging for GAN operations."""
    log_filename = f'gan_{datetime.now():%Y%m%d_%H%M%S}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_db_connection() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create database connection with proper timeout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'lotofacil.db')
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    return conn, cursor

def load_gan_config(cursor: sqlite3.Cursor) -> Dict[str, Any]:
    """Load GAN configuration from database."""
    cursor.execute('SELECT * FROM gan_configs ORDER BY id DESC LIMIT 1')
    config = cursor.fetchone()
    
    return {
        'latent_dim': config[3],
        'temperature': config[4],
        'dropout_rate': config[5],
        'learning_rate': config[6],
        'adam_beta1': config[7],
        'adam_beta2': config[8],
        'num_epochs': config[9],
        'batch_size': config[10],
        'test_size': config[11],
        'num_samples': config[12],
        'distance_metric': config[13],
        'gen_layer1': config[14],
        'gen_layer2': config[15],
        'gen_layer3': config[16],
        'disc_layer1': config[17],
        'disc_layer2': config[18],
        'disc_layer3': config[19],
        'training_seed': config[20]
    }

def load_training_data(cursor: sqlite3.Cursor, test_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load and split training data from draws table."""
    cursor.execute('''
        SELECT num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, 
               num_9, num_10, num_11, num_12, num_13, num_14, num_15
        FROM draws 
        ORDER BY draw_number
    ''')
    draws = cursor.fetchall()
    
    # Convert to numpy array and normalize
    data = np.array(draws)
    test_data = data[-test_size:]
    train_data = data[:-test_size]
    
    # Convert to tensor and normalize
    train_tensor = torch.FloatTensor(train_data)
    test_tensor = torch.FloatTensor(test_data)
    
    # Normalize
    train_tensor = (train_tensor - 1) / 24.0
    test_tensor = (test_tensor - 1) / 24.0
    
    return train_tensor, test_tensor

class Generator(nn.Module):
    def __init__(self, config: Dict[str, Any]):
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
    def __init__(self, config: Dict[str, Any]):
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

def train_gan(generator, discriminator, dataloader, config: Dict[str, Any], logger: logging.Logger):
    criterion = nn.BCELoss()
    g_optimizer = optim.Adam(
        generator.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta1'], config['adam_beta2'])
    )
    d_optimizer = optim.Adam(
        discriminator.parameters(), 
        lr=config['learning_rate'], 
        betas=(config['adam_beta1'], config['adam_beta2'])
    )
    
    generator.train()
    discriminator.train()
    
    for epoch in range(config['num_epochs']):
        for i, real_arrays in enumerate(dataloader):
            batch_size = real_arrays.size(0)
            
            # Train Discriminator
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
            
            # Train Generator
            g_optimizer.zero_grad()
            outputs = discriminator(fake_arrays)
            g_loss = criterion(outputs, real_labels)
            g_loss.backward()
            g_optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            logger.info(f'Epoch [{epoch+1}/{config["num_epochs"]}]')

def verify_unique_numbers(arrays: np.ndarray) -> bool:
    """Verify generated numbers are unique and valid."""
    for i, arr in enumerate(arrays):
        unique_nums = set(arr)
        if len(unique_nums) != 15:
            return False
        if not all(1 <= x <= 25 for x in arr):
            return False
    return True

def calculate_similarity_scores(generated_arrays: np.ndarray, test_set: np.ndarray, 
                              metric: str='euclidean') -> np.ndarray:
    """Calculate similarity scores using specified metric."""
    generated_sorted = np.sort(generated_arrays, axis=1)
    test_sorted = np.sort(test_set, axis=1)
    
    if metric == 'euclidean':
        distances = euclidean_distances(generated_sorted, test_sorted)
    elif metric == 'manhattan':
        distances = manhattan_distances(generated_sorted, test_sorted)
    elif metric == 'hamming':
        distances = cdist(generated_sorted, test_sorted, metric='hamming')
    else:
        raise ValueError(f"Métrica {metric} não suportada")
    
    min_distances = np.min(distances, axis=1)
    
    if metric == 'hamming':
        similarity_scores = 1 - min_distances
    else:
        similarity_scores = 1 / (1 + min_distances)
    
    return similarity_scores

def save_predictions(cursor: sqlite3.Cursor, conn: sqlite3.Connection, 
                    predictions: np.ndarray, scores: np.ndarray):
    """Save generated predictions to database."""
    # Get the maximum seq_order currently in the database
    cursor.execute('SELECT COALESCE(MAX(seq_order), -1) FROM predictions')
    max_seq_order = cursor.fetchone()[0]
    start_seq = max_seq_order + 1
    
    print(f"Starting sequence order from: {start_seq}")
    
    for idx, (pred, score) in enumerate(zip(predictions, scores)):
        cursor.execute('''
        INSERT INTO predictions (
            seq_order, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8,
            num_9, num_10, num_11, num_12, num_13, num_14, num_15, proximity
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', [start_seq + idx] + sorted(pred.tolist()) + [float(score)])
        
        if (idx + 1) % 1000 == 0:
            print(f"Saved {idx + 1} predictions...")
    
    conn.commit()
    print("All predictions saved successfully")

def generate_predictions(generator: nn.Module, test_tensor: torch.Tensor, 
                        config: Dict[str, Any], logger: logging.Logger) -> Tuple[np.ndarray, np.ndarray]:
    """Generate and evaluate predictions."""
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(config['num_samples'], config['latent_dim'])
        generated = generator(noise)
        final_arrays = denormalize_and_discretize(generated)
        samples = final_arrays.numpy().astype(int)
        
        if not verify_unique_numbers(samples):
            logger.error("Generated samples contain invalid or duplicate numbers")
            return None, None
        
        test_arrays = denormalize_and_discretize(test_tensor).numpy()
        similarity_scores = calculate_similarity_scores(
            samples, test_arrays, metric=config['distance_metric']
        )
        
        # Sort by similarity score
        sorted_indices = np.argsort(similarity_scores)[::-1]
        return samples[sorted_indices], similarity_scores[sorted_indices]

def denormalize_and_discretize(tensor: torch.Tensor) -> torch.Tensor:
    """Denormalize and round tensor values."""
    denorm = tensor * 24.0 + 1.0
    return torch.round(denorm).clamp(1, 25)

def main():
    logger = setup_logging()
    logger.info("Starting GAN training and prediction generation")
    
    try:
        conn, cursor = get_db_connection()
        
        # Load configuration
        config = load_gan_config(cursor)
        logger.info("Loaded GAN configuration from database")
        
        # Set random seeds
        torch.manual_seed(config['training_seed'])
        np.random.seed(config['training_seed'])
        
        # Load and prepare data
        train_tensor, test_tensor = load_training_data(cursor, config['test_size'])
        logger.info(f"Loaded {len(train_tensor)} training samples and {len(test_tensor)} test samples")
        
        # Create dataset and dataloader
        dataset = IntegerArrayDataset(train_tensor)
        dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
        
        # Initialize models
        generator = Generator(config)
        discriminator = Discriminator(config)
        
        # Train
        logger.info("Starting model training...")
        start_time = time.time()
        
        train_gan(generator, discriminator, dataloader, config, logger)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f} seconds")
        
        # Generate predictions
        logger.info(f"Generating {config['num_samples']} predictions...")
        start_time = time.time()
        
        predictions, scores = generate_predictions(generator, test_tensor, config, logger)
        if predictions is not None:
            save_predictions(cursor, conn, predictions, scores)
            
            generation_time = time.time() - start_time
            logger.info(f"Generation and saving completed in {generation_time:.2f} seconds")
            
            logger.info("\nTop 5 predictions by similarity score:")
            for i in range(min(5, len(predictions))):
                logger.info(f"Prediction {i+1}: {sorted(predictions[i])} (Score: {scores[i]:.4f})")
        else:
            logger.error("Failed to generate valid predictions")
        
    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    main()