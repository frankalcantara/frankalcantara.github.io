import csv
import sqlite3
from pathlib import Path

def create_database():
    """
    Creates a new SQLite database with a fresh palpites table.
    First removes any existing table to ensure we have the correct structure.
    
    Returns:
        tuple: A pair of (connection, cursor) objects for database operations
    """
    # Create a new database connection with a reasonable timeout
    conn = sqlite3.connect('lotofacil.db', timeout=30)
    cursor = conn.cursor()
    
    # First, drop the existing table if it exists
    cursor.execute('DROP TABLE IF EXISTS palpites')
    
    # Now create the table with our new structure including seq_order
    # We create all fields explicitly for clarity and maintainability
    create_table_sql = '''
    CREATE TABLE palpites (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        seq_order INTEGER UNIQUE,
        num_1 INTEGER, num_2 INTEGER, num_3 INTEGER, num_4 INTEGER, num_5 INTEGER,
        num_6 INTEGER, num_7 INTEGER, num_8 INTEGER, num_9 INTEGER, num_10 INTEGER,
        num_11 INTEGER, num_12 INTEGER, num_13 INTEGER, num_14 INTEGER, num_15 INTEGER,
        tipo CHAR(1)
    )
    '''
    
    cursor.execute(create_table_sql)
    conn.commit()
    
    print("Database structure created successfully")
    return conn, cursor

def process_csv(conn, cursor, csv_file):
    """
    Reads the CSV file and inserts data into SQLite database.
    Skips the header row if present and adds sequence numbers starting from 0.
    
    Args:
        conn: SQLite connection object for committing transactions
        cursor: SQLite cursor object for executing queries
        csv_file: Path to the CSV file to process
    """
    insert_sql = '''
    INSERT INTO palpites (
        seq_order, 
        num_1, num_2, num_3, num_4, num_5, 
        num_6, num_7, num_8, num_9, num_10, 
        num_11, num_12, num_13, num_14, num_15, 
        tipo
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    '''
    
    with open(csv_file, 'r') as file:
        # Read first line to check if it's a header
        first_line = file.readline().strip()
        file.seek(0)  # Go back to start of file
        
        csv_reader = csv.reader(file)
        
        # Check if first row is header (contains non-numeric data)
        try:
            first_row = [int(x) for x in next(csv_reader)]
            # If we get here, first row was numeric, go back to start
            file.seek(0)
            csv_reader = csv.reader(file)
        except ValueError:
            # First row wasn't numeric, it was a header - continue from next row
            pass
        
        # Process each data row
        rows_processed = 0
        for seq_order, row in enumerate(csv_reader):
            try:
                # Verify we have exactly 15 numbers
                if len(row) != 15:
                    print(f"Warning: Row {seq_order} has {len(row)} values instead of 15, skipping")
                    continue
                    
                # Convert string numbers to integers, add seq_order and 'F'
                values = [seq_order] + [int(num) for num in row] + ['F']
                cursor.execute(insert_sql, values)
                rows_processed += 1
                
                # Commit every 5000 rows to avoid huge transactions
                if rows_processed % 5000 == 0:
                    print(f"Processed {rows_processed} rows...")
                    conn.commit()
                    
            except ValueError as e:
                print(f"Warning: Invalid data in row {seq_order}: {row}")
                print(f"Error details: {str(e)}")
                continue

def main():
    """
    Main function that orchestrates the CSV to SQLite conversion process.
    Handles database connection, file processing, and error reporting.
    """
    csv_file = 'gan_generated_lotofacil.csv'
    conn = None
    
    try:
        # Verify file existence before starting
        if not Path(csv_file).exists():
            print(f"Error: File {csv_file} not found")
            return
        
        print("Starting database creation...")
        conn, cursor = create_database()
        
        print("Beginning CSV processing...")
        process_csv(conn, cursor, csv_file)
        
        # Final commit to ensure all data is saved
        conn.commit()
        
        # Generate and display processing summary
        cursor.execute('SELECT COUNT(*), MIN(seq_order), MAX(seq_order) FROM palpites')
        count, min_order, max_order = cursor.fetchone()
        print(f"\nProcessing Summary:")
        print(f"- Total rows successfully processed: {count}")
        print(f"- Sequence order range: {min_order} to {max_order}")
        
        # Verify data structure
        cursor.execute('SELECT * FROM palpites LIMIT 1')
        columns = [description[0] for description in cursor.description]
        print(f"\nTable structure verification:")
        print(f"- Columns: {', '.join(columns)}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        if conn:
            print("Rolling back any incomplete changes...")
            conn.rollback()
    finally:
        if conn:
            conn.close()
            print("Database connection closed")

if __name__ == '__main__':
    main()