#!/usr/bin/env python3
import sqlite3
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

def get_password():
    """
    Read Gmail password from the first line of gmail.sec file
    
    Returns:
        str: The password string with whitespace removed
        
    Raises:
        Exception: If file is not found or cannot be read
    """
    try:
        with open('gmail.sec') as f:
            return f.readline().strip()
    except FileNotFoundError:
        raise Exception("Arquivo gmail.sec não encontrado")
    except Exception as e:
        raise Exception(f"Erro ao ler senha: {str(e)}")

def get_lottery_numbers(cursor):
    """
    Retrieve first 10 unprocessed rows from the database ordered by seq_order
    
    Args:
        cursor: SQLite cursor object
    
    Returns:
        list: List of tuples containing lottery numbers
    """
    # Select records where tipo = 'F', ordered by seq_order
    cursor.execute("""
        SELECT seq_order, num_1, num_2, num_3, num_4, num_5, num_6, num_7, 
               num_8, num_9, num_10, num_11, num_12, num_13, num_14, num_15
        FROM palpites 
        WHERE tipo = 'F'
        ORDER BY seq_order
        LIMIT 10
    """)
    return cursor.fetchall()

def update_status(cursor, conn, processed_orders):
    """
    Update tipo to 'V' for processed records
    
    Args:
        cursor: SQLite cursor object
        conn: SQLite connection object
        processed_orders: List of seq_order values to update
    """
    # Convert sequence orders to string for SQL IN clause
    orders_str = ','.join(map(str, processed_orders))
    
    cursor.execute(f"""
        UPDATE palpites 
        SET tipo = 'V' 
        WHERE seq_order IN ({orders_str})
    """)
    conn.commit()

def process_and_send_email(db_path, email):
    """
    Process lottery data from SQLite database and send results via email
    
    Args:
        db_path (str): Path to the SQLite database
        email (str): Gmail address to use for sending
    """
    try:
        # Establish database connection
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Get lottery numbers
        rows = get_lottery_numbers(cursor)
        
        if not rows:
            print("Não há mais números para processar")
            conn.close()
            return
        
        # Store seq_order values for updating status later
        processed_orders = [row[0] for row in rows]
        
        # Convert rows to space-separated strings, excluding seq_order
        selected_rows = "\n".join(
            [" ".join(map(str, row[1:])) for row in rows]
        )
        
        # Configure email message
        msg = MIMEMultipart()
        msg['From'] = email
        msg['To'] = 'frank.alcantara@gmail.com'
        msg['Subject'] = 'Dados Lotofácil'
        
        # Add lottery numbers to email body
        body = f"Segue os arrays:\n\n{selected_rows}"
        msg.attach(MIMEText(body, 'plain'))
        
        # Get password and establish secure SMTP connection
        password = get_password()
        server = smtplib.SMTP('smtp.gmail.com', 587)
        server.starttls()
        server.login(email, password)
        
        # Send the email
        text = msg.as_string()
        server.sendmail(email, 'frank.alcantara@gmail.com', text)
        server.quit()
        print("Email enviado com sucesso!")
        
        # Update status flags in the database
        update_status(cursor, conn, processed_orders)
        print(f"Status atualizado para {len(processed_orders)} registros (seq_order de {min(processed_orders)} a {max(processed_orders)})")
        
    except Exception as e:
        print(f"Erro ao processar dados: {str(e)}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    db_path = "lotofacil.db"
    email = "frank.alcantara@gmail.com"
    
    process_and_send_email(db_path, email)