#!/usr/bin/env python3
import sqlite3
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

def setup_logging() -> logging.Logger:
    """Configure logging for database initialization."""
    log_filename = f'db_init_{datetime.now():%Y%m%d_%H%M%S}.log'
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_db_path() -> str:
    """Get the absolute path for the database file."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(script_dir, 'lotofacil.db')

def create_connection() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create database connection with extended timeout."""
    db_path = get_db_path()
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    return conn, cursor

def create_gan_configs_table(cursor: sqlite3.Cursor, logger: logging.Logger):
    """Create the GAN configurations table and insert default values."""
    logger.info("Creating gan_configs table...")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS gan_configs (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER NOT NULL,
        modified_at TEXT NOT NULL,
        latent_dim INTEGER NOT NULL,
        temperature REAL NOT NULL,
        dropout_rate REAL NOT NULL, 
        learning_rate REAL NOT NULL,
        adam_beta1 REAL NOT NULL,
        adam_beta2 REAL NOT NULL,
        num_epochs INTEGER NOT NULL,
        batch_size INTEGER NOT NULL,
        test_size INTEGER NOT NULL,
        num_samples INTEGER NOT NULL,
        distance_metric TEXT NOT NULL,
        gen_layer1 INTEGER NOT NULL,
        gen_layer2 INTEGER NOT NULL,
        gen_layer3 INTEGER NOT NULL,
        disc_layer1 INTEGER NOT NULL,
        disc_layer2 INTEGER NOT NULL,
        disc_layer3 INTEGER NOT NULL,
        training_seed INTEGER NOT NULL
    )''')
    
    # Insert default configuration if not exists
    cursor.execute('''
    INSERT OR IGNORE INTO gan_configs (
        user_id, modified_at, latent_dim, temperature, dropout_rate,
        learning_rate, adam_beta1, adam_beta2, num_epochs, batch_size,
        test_size, num_samples, distance_metric, gen_layer1, gen_layer2,
        gen_layer3, disc_layer1, disc_layer2, disc_layer3, training_seed
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (
        1,                          # user_id
        datetime.now().isoformat(), # modified_at
        50,                         # latent_dim 
        0.5,                        # temperature
        0.3,                        # dropout_rate
        0.0002,                     # learning_rate
        0.5,                        # adam_beta1
        0.999,                      # adam_beta2
        150,                        # num_epochs
        32,                         # batch_size
        100,                        # test_size
        50000,                      # num_samples
        'manhattan',                # distance_metric
        128,                        # gen_layer1
        256,                        # gen_layer2
        128,                        # gen_layer3
        128,                        # disc_layer1
        256,                        # disc_layer2
        128,                        # disc_layer3
        int(time.time_ns()) % (2**32)  # training_seed
    ))
    
    logger.info("gan_configs table created and initialized with default values")

def create_predictions_table(cursor: sqlite3.Cursor, logger: logging.Logger):
    """Create the predictions table with improved structure."""
    logger.info("Creating predictions table...")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seq_order INTEGER UNIQUE,
        num_1 INTEGER NOT NULL,
        num_2 INTEGER NOT NULL,
        num_3 INTEGER NOT NULL,
        num_4 INTEGER NOT NULL,
        num_5 INTEGER NOT NULL,
        num_6 INTEGER NOT NULL,
        num_7 INTEGER NOT NULL,
        num_8 INTEGER NOT NULL,
        num_9 INTEGER NOT NULL,
        num_10 INTEGER NOT NULL,
        num_11 INTEGER NOT NULL,
        num_12 INTEGER NOT NULL,
        num_13 INTEGER NOT NULL,
        num_14 INTEGER NOT NULL,
        num_15 INTEGER NOT NULL,
        status CHAR(1) NOT NULL DEFAULT 'F',
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        processed_at TEXT,
        matches INTEGER,
        proximity REAL,
        CONSTRAINT valid_numbers CHECK (
            num_1 BETWEEN 1 AND 25 AND
            num_2 BETWEEN 1 AND 25 AND
            num_3 BETWEEN 1 AND 25 AND
            num_4 BETWEEN 1 AND 25 AND
            num_5 BETWEEN 1 AND 25 AND
            num_6 BETWEEN 1 AND 25 AND
            num_7 BETWEEN 1 AND 25 AND
            num_8 BETWEEN 1 AND 25 AND
            num_9 BETWEEN 1 AND 25 AND
            num_10 BETWEEN 1 AND 25 AND
            num_11 BETWEEN 1 AND 25 AND
            num_12 BETWEEN 1 AND 25 AND
            num_13 BETWEEN 1 AND 25 AND
            num_14 BETWEEN 1 AND 25 AND
            num_15 BETWEEN 1 AND 25
        ),
        CONSTRAINT valid_status CHECK (status IN ('F', 'P', 'V'))
    )''')
    
    # Index for faster queries on status and matches
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_predictions_status 
    ON predictions(status)
    ''')
    
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_predictions_matches 
    ON predictions(matches)
    ''')
    
    logger.info("predictions table created successfully")

def create_draws_table(cursor: sqlite3.Cursor, logger: logging.Logger):
    """Create table for storing official lottery draws."""
    logger.info("Creating draws table...")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS draws (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        draw_date TEXT NOT NULL,
        draw_number INTEGER UNIQUE NOT NULL,
        num_1 INTEGER NOT NULL,
        num_2 INTEGER NOT NULL,
        num_3 INTEGER NOT NULL,
        num_4 INTEGER NOT NULL,
        num_5 INTEGER NOT NULL,
        num_6 INTEGER NOT NULL,
        num_7 INTEGER NOT NULL,
        num_8 INTEGER NOT NULL,
        num_9 INTEGER NOT NULL,
        num_10 INTEGER NOT NULL,
        num_11 INTEGER NOT NULL,
        num_12 INTEGER NOT NULL,
        num_13 INTEGER NOT NULL,
        num_14 INTEGER NOT NULL,
        num_15 INTEGER NOT NULL,
        created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
        CONSTRAINT valid_draw_numbers CHECK (
            num_1 BETWEEN 1 AND 25 AND
            num_2 BETWEEN 1 AND 25 AND
            num_3 BETWEEN 1 AND 25 AND
            num_4 BETWEEN 1 AND 25 AND
            num_5 BETWEEN 1 AND 25 AND
            num_6 BETWEEN 1 AND 25 AND
            num_7 BETWEEN 1 AND 25 AND
            num_8 BETWEEN 1 AND 25 AND
            num_9 BETWEEN 1 AND 25 AND
            num_10 BETWEEN 1 AND 25 AND
            num_11 BETWEEN 1 AND 25 AND
            num_12 BETWEEN 1 AND 25 AND
            num_13 BETWEEN 1 AND 25 AND
            num_14 BETWEEN 1 AND 25 AND
            num_15 BETWEEN 1 AND 25
        )
    )''')
    
    cursor.execute('''
    CREATE INDEX IF NOT EXISTS idx_draws_date 
    ON draws(draw_date)
    ''')
    
    logger.info("draws table created successfully")

def init_database():
    """Initialize the complete database structure."""
    logger = setup_logging()
    logger.info("Starting database initialization...")
    
    try:
        conn, cursor = create_connection()
        
        # Create all tables
        create_gan_configs_table(cursor, logger)
        create_predictions_table(cursor, logger)
        create_draws_table(cursor, logger)
        
        # Commit changes
        conn.commit()
        logger.info("Database initialization completed successfully")
        
    except sqlite3.Error as e:
        logger.error(f"SQLite error occurred: {e}")
        if conn:
            conn.rollback()
        raise
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()
            logger.info("Database connection closed")

if __name__ == "__main__":
    init_database()