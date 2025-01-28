#!/usr/bin/env python3
import sqlite3
import os
from typing import Tuple

def get_db_connection() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create database connection with proper timeout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'lotofacil.db')
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    return conn, cursor

def remove_duplicates():
    """Remove duplicate predictions keeping only the first occurrence."""
    conn = None
    try:
        conn, cursor = get_db_connection()
        print("Conectado ao banco de dados")
        
        # Count total records before
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_before = cursor.fetchone()[0]
        print(f"Total de registros antes: {total_before}")
        
        # Create temporary table with unique combinations
        cursor.execute('''
            CREATE TEMP TABLE temp_predictions AS
            SELECT MIN(id) as min_id, 
                   num_1, num_2, num_3, num_4, num_5,
                   num_6, num_7, num_8, num_9, num_10,
                   num_11, num_12, num_13, num_14, num_15
            FROM predictions
            GROUP BY num_1, num_2, num_3, num_4, num_5,
                     num_6, num_7, num_8, num_9, num_10,
                     num_11, num_12, num_13, num_14, num_15
        ''')
        
        # Delete duplicates
        cursor.execute('''
            DELETE FROM predictions 
            WHERE id NOT IN (SELECT min_id FROM temp_predictions)
        ''')
        
        # Drop temporary table
        cursor.execute('DROP TABLE temp_predictions')
        
        # Count remaining records
        cursor.execute('SELECT COUNT(*) FROM predictions')
        total_after = cursor.fetchone()[0]
        
        # Commit changes
        conn.commit()
        
        duplicates_removed = total_before - total_after
        print(f"Total de registros após: {total_after}")
        print(f"Duplicatas removidas: {duplicates_removed}")
        
    except Exception as e:
        print(f"Erro ao remover duplicatas: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Conexão com o banco fechada")

if __name__ == "__main__":
    remove_duplicates()