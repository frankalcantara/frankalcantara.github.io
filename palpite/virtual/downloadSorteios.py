#!/usr/bin/env python3
import requests
import pandas as pd
import io
from datetime import datetime
import os
from pathlib import Path
import sqlite3
import logging
from typing import Tuple, Optional

def setup_logging() -> logging.Logger:
    """Configure logging for download operations."""
    log_filename = f'download_{datetime.now():%Y%m%d_%H%M%S}.log'
    
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

def get_last_draw_number(cursor: sqlite3.Cursor) -> Optional[int]:
    """Get the last draw number from the database."""
    cursor.execute('SELECT MAX(draw_number) FROM draws')
    result = cursor.fetchone()[0]
    return result if result is not None else 0

def save_to_database(df: pd.DataFrame, conn: sqlite3.Connection, cursor: sqlite3.Cursor, logger: logging.Logger) -> bool:
    """Save new draws to the database."""
    try:
        last_draw = get_last_draw_number(cursor)
        new_records = 0
        
        for _, row in df.iterrows():
            draw_number = row.iloc[0]  # Assuming first column is draw number
            if draw_number > last_draw:
                draw_date = row.iloc[1]  # Assuming second column is date
                numbers = row.iloc[2:17].values  # Get the 15 numbers
                
                cursor.execute('''
                INSERT INTO draws (
                    draw_date, draw_number, 
                    num_1, num_2, num_3, num_4, num_5,
                    num_6, num_7, num_8, num_9, num_10,
                    num_11, num_12, num_13, num_14, num_15
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', [draw_date, draw_number] + numbers.tolist())
                new_records += 1
        
        conn.commit()
        logger.info(f"Added {new_records} new draws to database")
        return new_records > 0
        
    except Exception as e:
        logger.error(f"Error saving to database: {e}")
        conn.rollback()
        return False

def check_and_download_lotofacil():
    """Download latest lottery results and save to both CSV and database."""
    logger = setup_logging()
    url = "https://jolly-flower-cfbe.frank-alcantara.workers.dev/"
    output_csv = 'lotofacil.csv'
    
    def get_last_saved_date():
        try:
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                last_date = pd.to_datetime(df.iloc[-1, 0])
                return last_date
            return None
        except Exception as e:
            logger.error(f"Error reading last saved date: {e}")
            return None
    
    try:
        # Get last saved date
        last_saved_date = get_last_saved_date()
        
        # Download current file
        response = requests.get(url)
        response.raise_for_status()
        
        # Read Excel file
        current_df = pd.read_excel(
            io.BytesIO(response.content),
            usecols="A:Q",  # Modified to include draw number
            skiprows=1
        )
        
        # Get the last date from current file
        current_last_date = pd.to_datetime(current_df.iloc[-1, 1], dayfirst=True)  # Parse date in DD/MM/YYYY format
        
        update_needed = last_saved_date is None or current_last_date > last_saved_date
        
        if update_needed:
            # Save to CSV
            current_df.to_csv(output_csv, index=False)
            logger.info(f"New data saved to CSV. Last date: {current_last_date.strftime('%Y-%m-%d')}")
            
            # Save to database
            conn, cursor = get_db_connection()
            try:
                if save_to_database(current_df, conn, cursor, logger):
                    logger.info("Database updated successfully")
                else:
                    logger.info("No new records to add to database")
            finally:
                conn.close()
            
            return True
        else:
            logger.info(f"No new data available. Last saved date: {last_saved_date.strftime('%Y-%m-%d')}")
            return False
            
    except requests.RequestException as e:
        logger.error(f"Error downloading the file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error processing the file: {e}")
        return False

if __name__ == "__main__":
    # Remove existing CSV file before downloading new one
    if os.path.exists('lotofacil.csv'):
        os.remove('lotofacil.csv')
    
    check_and_download_lotofacil()