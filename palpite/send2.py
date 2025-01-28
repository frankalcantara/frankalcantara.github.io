#!/usr/bin/env python3
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os
from datetime import datetime
from typing import List, Tuple, Optional

def get_db_connection() -> Tuple[sqlite3.Connection, sqlite3.Cursor]:
    """Create database connection with proper timeout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    db_path = os.path.join(script_dir, 'lotofacil.db')
    conn = sqlite3.connect(db_path, timeout=30)
    cursor = conn.cursor()
    return conn, cursor

def get_password() -> str:
    """Read Gmail password from the first line of gmail.sec file."""
    try:
        with open('/home/frankalcantara.github.io/palpite/virtual/gmail.sec') as f:
            return f.readline().strip()
    except FileNotFoundError:
        raise Exception("Arquivo gmail.sec não encontrado")
    except Exception as e:
        raise Exception(f"Erro ao ler senha: {str(e)}")

def get_predictions_to_send(cursor: sqlite3.Cursor, limit: int = 10) -> List[tuple]:
    """Retrieve predictions to be sent, ordered by matches (highest first) and proximity."""
    cursor.execute('''
        SELECT 
            num_1, num_2, num_3, num_4, num_5, 
            num_6, num_7, num_8, num_9, num_10,
            num_11, num_12, num_13, num_14, num_15,
            matches
        FROM predictions 
        WHERE status = 'V' 
          AND matches >= 11
          AND processed_at >= DATE('now', '-1 day')
        ORDER BY matches DESC, proximity DESC
        LIMIT ?
    ''', (limit,))
    return cursor.fetchall()

def mark_predictions_as_sent(cursor: sqlite3.Cursor, conn: sqlite3.Connection):
    """Mark processed predictions as sent."""
    cursor.execute('''
        UPDATE predictions
        SET status = 'P'
        WHERE status = 'V' 
          AND matches >= 11
          AND processed_at >= DATE('now', '-1 day')
    ''')
    conn.commit()

def format_email_body(predictions: List[tuple]) -> str:
    """Format predictions into email body."""
    if not predictions:
        return "Não há previsões para enviar hoje."
    
    body = "Previsões para Lotofácil:\n\n"
    for pred in predictions:
        numbers = pred[:15]  # First 15 items are the numbers
        matches = pred[15]   # Last item is the match count
        sorted_numbers = ' '.join(map(str, sorted(numbers)))
        body += f"Números: {sorted_numbers} (Acertos anteriores: {matches})\n"
    
    return body

def send_email(predictions: List[tuple], email: str, password: str):
    """Send email with predictions."""
    msg = MIMEMultipart()
    msg['From'] = email
    msg['To'] = 'frank.alcantara@gmail.com'
    msg['Subject'] = f'Lotofácil - Previsões {datetime.now():%d/%m/%Y}'
    
    body = format_email_body(predictions)
    msg.attach(MIMEText(body, 'plain'))
    
    # Establish secure SMTP connection
    server = smtplib.SMTP('smtp.gmail.com', 587)
    try:
        server.starttls()
        server.login(email, password)
        
        # Send email
        server.sendmail(email, 'frank.alcantara@gmail.com', msg.as_string())
        print("Email enviado com sucesso!")
        
    except Exception as e:
        print(f"Erro ao enviar email: {str(e)}")
        raise
    finally:
        server.quit()

def process_and_send_email():
    """Main function to process predictions and send email."""
    email = "frank.alcantara@gmail.com"
    conn = None
    
    try:
        # Get database connection
        conn, cursor = get_db_connection()
        print("Conectado ao banco de dados")
        
        # Get predictions to send
        predictions = get_predictions_to_send(cursor)
        print(f"Encontradas {len(predictions)} previsões para enviar")
        
        if predictions:
            # Get password and send email
            password = get_password()
            send_email(predictions, email, password)
            
            # Mark predictions as sent
            mark_predictions_as_sent(cursor, conn)
            print("Status das previsões atualizado no banco")
        else:
            print("Não há previsões para enviar")
        
    except Exception as e:
        print(f"Erro ao processar dados: {str(e)}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Conexão com o banco fechada")

if __name__ == "__main__":
    process_and_send_email()