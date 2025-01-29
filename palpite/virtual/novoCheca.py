#!/usr/bin/env python3
import sqlite3
from datetime import datetime
import os
from pathlib import Path
import numpy as np

def get_db_connection():
    """Create database connection with proper timeout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'lotofacil.db')
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    return conn, cursor

def get_last_draw(cursor):
    """Get the most recent lottery draw."""
    cursor.execute('''
    SELECT num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8,
           num_9, num_10, num_11, num_12, num_13, num_14, num_15
    FROM draws 
    ORDER BY draw_number DESC 
    LIMIT 1
    ''')
    return cursor.fetchone()

def count_matches(real_numbers, prediction_numbers):
    """Count matching numbers between a prediction and actual draw."""
    return sum(np.isin(real_numbers, prediction_numbers))

def analyze_predictions(cursor, conn, last_draw):
    """Analyze all unprocessed predictions against last draw."""
    matches_count = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    
    cursor.execute('''
    SELECT id, num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8,
           num_9, num_10, num_11, num_12, num_13, num_14, num_15
    FROM predictions 
    WHERE status = 'F'
    ''')
    predictions = cursor.fetchall()
    
    print(f"\nAnalyzing {len(predictions)} predictions...")
    
    for pred in predictions:
        pred_id = pred[0]
        numbers = pred[1:]
        match_count = count_matches(last_draw, numbers)
        
        if match_count >= 11:
            matches_count[match_count] += 1
        
        # Update prediction with match count and mark as verified
        cursor.execute('''
        UPDATE predictions 
        SET status = 'V', 
            matches = ?,
            processed_at = CURRENT_TIMESTAMP
        WHERE id = ?
        ''', (match_count, pred_id))
    
    conn.commit()
    return matches_count

def get_gan_version():
    """Get GAN.py modification time."""
    try:
        # Try local directory first
        script_dir = os.path.dirname(os.path.abspath(__file__))
        gan_path = os.path.join(script_dir, 'GAN.py')
        
        if not os.path.exists(gan_path):
            # Try virtual directory if local doesn't exist
            gan_path = '/home/frankalcantara.github.io/palpite/virtual/GAN.py'
            
        if os.path.exists(gan_path):
            return datetime.fromtimestamp(os.path.getmtime(gan_path)).strftime('%Y%m%d')
        else:
            print("Warning: GAN.py not found, using current date as version")
            return datetime.now().strftime('%Y%m%d')
            
    except Exception as e:
        print(f"Warning: Error getting GAN version: {e}")
        return datetime.now().strftime('%Y%m%d')

def get_last_ten_results(cursor):
    """Get the last 10 execution results."""
    cursor.execute('''
    SELECT 
        DATE(processed_at) as exec_date,
        COUNT(CASE WHEN matches = 11 THEN 1 END) as matches_11,
        COUNT(CASE WHEN matches = 12 THEN 1 END) as matches_12,
        COUNT(CASE WHEN matches = 13 THEN 1 END) as matches_13,
        COUNT(CASE WHEN matches = 14 THEN 1 END) as matches_14,
        COUNT(CASE WHEN matches = 15 THEN 1 END) as matches_15
    FROM predictions
    WHERE status = 'V'
    GROUP BY DATE(processed_at)
    ORDER BY processed_at DESC
    LIMIT 10
    ''')
    return cursor.fetchall()

def generate_html_table(results, gan_version):
    """Generate complete HTML file with results table."""
    html_content = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
    <meta charset="UTF-8">
    <title>Resultados GAN Lotofácil</title>
    <style>
        table { 
            border-collapse: collapse; 
            width: 100%; 
            font-family: Arial, sans-serif;
        }
        th, td { 
            border: 1px solid black; 
            padding: 8px; 
            text-align: center; 
        }
        th {
            background-color: #f2f2f2;
        }
        tr:nth-child(even) {
            background-color: #f9f9f9;
        }
        tr:hover {
            background-color: #f5f5f5;
        }
    </style>
</head>
<body>
    <table>
        <tr>
            <th>Data Execução</th>
            <th>Versão GAN</th>
            <th>11 Acertos</th>
            <th>12 Acertos</th>
            <th>13 Acertos</th>
            <th>14 Acertos</th>
            <th>15 Acertos</th>
        </tr>
"""
    
    # Add data rows
    for result in results:
        exec_date = result[0]
        matches = result[1:]
        row = f"""
        <tr>
            <td>{exec_date}</td>
            <td>{gan_version}</td>
            <td>{matches[0]}</td>
            <td>{matches[1]}</td>
            <td>{matches[2]}</td>
            <td>{matches[3]}</td>
            <td>{matches[4]}</td>
        </tr>"""
        html_content += row
    
    html_content += """
    </table>
</body>
</html>
"""
    
    # Write to both locations
    locations = [
        '/var/www/html/table.html',
        '/home/frankalcantara.github.io/assets/table.html'
    ]
    
    for location in locations:
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, 'w') as f:
            f.write(html_content)
        os.chmod(location, 0o644)
    
    print("HTML table updated successfully")

def main():
    try:
        print("Starting analysis...")
        conn, cursor = get_db_connection()
        
        # Get last draw
        last_draw = get_last_draw(cursor)
        print(f"Last draw numbers: {last_draw}")
        
        # Analyze predictions
        matches = analyze_predictions(cursor, conn, last_draw)
        
        # Print results
        print("\nAnalysis Results:")
        for match_count, total in matches.items():
            print(f"{match_count} matches: {total} occurrences")
        
        # Get GAN version
        gan_version = get_gan_version()
        
        # Get last results and generate HTML
        results = get_last_ten_results(cursor)
        generate_html_table(results, gan_version)
        
        print("\nAnalysis completed successfully")
        
    except Exception as e:
        print(f"Error during analysis: {str(e)}")
        raise
    finally:
        if 'conn' in locals():
            conn.close()
            print("Database connection closed")

if __name__ == "__main__":
    main()