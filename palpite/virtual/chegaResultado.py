import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import os

def count_matches(real_numbers, generated_array):
    return sum(np.isin(real_numbers, generated_array))

def calculate_proximity(array, last_100_draws):
    matches = [count_matches(array, draw) for draw in last_100_draws]
    return np.mean(matches)

def save_fourteen_matches(arrays_14, output_file='quatorze.csv'):
    """
    Save arrays with 14 matches in the same format as lotofacil.csv
    """
    today = datetime.now().strftime('%Y-%m-%d')
    
    # Create DataFrame with date and numbers columns
    fourteen_data = []
    for _, row in arrays_14.iterrows():
        numbers = [int(row[f'num_{i+1}']) for i in range(15)]
        fourteen_data.append({
            'Data': today,
            **{f'Bola{i+1}': num for i, num in enumerate(sorted(numbers))}
        })
    
    df_fourteen = pd.DataFrame(fourteen_data)
    
    # Save to CSV file
    if fourteen_data:
        df_fourteen.to_csv(output_file, mode='a', header=False, index=False)
        print(f"Saved {len(fourteen_data)} arrays with 14 matches to {output_file}")
    else:
        print("No arrays with 14 matches to save")

def check_web_directory():
    """Check and prepare web directory for writing"""
    web_dir = '/var/www/html'
    try:
        # Create directory if it doesn't exist
        if not os.path.exists(web_dir):
            os.makedirs(web_dir, exist_ok=True)
            print(f"Created directory: {web_dir}")
        
        # Check permissions
        if not os.access(web_dir, os.W_OK):
            print(f"Warning: No write permission for {web_dir}")
            return False
        return True
    except Exception as e:
        print(f"Error checking web directory: {e}")
        return False

def update_html_table(matches, lotofacil_file, gan_file='gan.py', table_file='/var/www/html/table.html'):
    """Update HTML table by adding a new row with current execution results"""
    try:
        timestamp = datetime.now().strftime('%d/%m/%Y')
        
        # Get GAN.py modification date
        gan_mod_time = datetime.fromtimestamp(os.path.getmtime(gan_file)).strftime('%Y%m%d')
        
        # Create base HTML if file doesn't exist
        if not os.path.exists(table_file):
            html_content = """
            <!DOCTYPE html>
            <html lang="pt-BR">
            <head>
                <meta charset="UTF-8">
                <title>Resultados GAN Lotofácil</title>
                <style>
                    table { border-collapse: collapse; width: 100%; }
                    th, td { border: 1px solid black; padding: 8px; text-align: center; }
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
                </table>
            </body>
            </html>
            """
            soup = BeautifulSoup(html_content, 'html.parser')
        else:
            # Read existing file
            with open(table_file, 'r') as f:
                soup = BeautifulSoup(f.read(), 'html.parser')

        # Create new row with current results
        table = soup.find('table')
        new_row = soup.new_tag('tr')
        
        # Add timestamp
        td = soup.new_tag('td')
        td.string = timestamp
        new_row.append(td)
        
        # Add GAN version
        td = soup.new_tag('td')
        td.string = gan_mod_time
        new_row.append(td)
        
        # Add match counts
        for i in range(11, 16):
            td = soup.new_tag('td')
            td.string = str(matches.get(i, 0))
            new_row.append(td)
        
        # Insert new row after header
        table.append(new_row)
        
        # Write updated HTML to file
        with open(table_file, 'w') as f:
            f.write(str(soup))
        
        # Try to set proper permissions if writing to web directory
        if table_file.startswith('/var/www/'):
            try:
                os.chmod(table_file, 0o644)
            except Exception as e:
                print(f"Warning: Could not set file permissions: {e}")
        
        print(f"HTML table updated successfully at {table_file}")
        
    except Exception as e:
        print(f"Error updating HTML table: {e}")
        # Don't raise the exception - allow the script to continue
        print("Continuing with the rest of the analysis...")
        
def analyze_gan_results(lotofacil_file, gan_file, output_file):
    # Read all data from lotofacil.csv
    df_real = pd.read_csv(lotofacil_file)
    last_draw = df_real.iloc[-1].values[1:16]  # Skip date, get 15 numbers
    last_100_draws = df_real.iloc[-100:].values[:, 1:16]  # Get last 100 draws
    
    # Read GAN generated numbers and remove duplicates
    df_gan = pd.read_csv(gan_file)
    df_gan = df_gan.drop_duplicates()
    print(f"Original GAN arrays: {len(df_gan)}")
    print(f"Unique GAN arrays after removing duplicates: {len(df_gan)}")
    
    # Initialize counters and storage for matches
    matches = {11: 0, 12: 0, 13: 0, 14: 0, 15: 0}
    selected_arrays = []
    
    # Compare each GAN generated row with the last real draw
    for idx, row in df_gan.iterrows():
        gan_numbers = row.values
        match_count = count_matches(last_draw, gan_numbers)
        if match_count >= 11:
            matches[match_count] += 1
            proximity = calculate_proximity(gan_numbers, last_100_draws)
            selected_arrays.append({
                'numbers': gan_numbers,
                'matches': match_count,
                'proximity': proximity
            })
    
    # Generate report
    print("Analysis Report")
    print("-" * 50)
    print(f"Last real draw: {last_draw}")
    print("\nMatches found in GAN generated numbers:")
    for match_count, total in matches.items():
        print(f"{match_count} numbers: {total} occurrences")
    
    # Sort selected arrays by proximity
    selected_arrays.sort(key=lambda x: x['proximity'], reverse=True)
    
    # Create DataFrame and save to file
    output_data = pd.DataFrame([{
        'matches': arr['matches'],
        'proximity': arr['proximity'],
        **{f'num_{i+1}': num for i, num in enumerate(arr['numbers'])}
    } for arr in selected_arrays])
    
    if not output_data.empty:
        output_data.to_csv(output_file, index=False)
        print(f"\nSelected arrays saved to {output_file}")
        print(f"Total arrays saved: {len(output_data)}")
        
        # Save arrays with 14 matches to quatorze.csv
        arrays_14 = output_data[output_data['matches'] == 14]
        save_fourteen_matches(arrays_14)
        
        # Print first 10 arrays
        print("\nTop 10 arrays by proximity to last 100 draws:")
        print("-" * 50)
        for idx, row in output_data.head(10).iterrows():
            numbers = [int(row[f'num_{i+1}']) for i in range(15)]
            print(f"Array {idx + 1}:")
            print(f"Numbers: {numbers}")
            print(f"Matches with last draw: {int(row['matches'])}")
            print(f"Average proximity: {row['proximity']:.2f}")
            print("-" * 30)
    
    # Update HTML table
    update_html_table(matches, lotofacil_file, 'GAN.py')

if __name__ == "__main__":
    analyze_gan_results(
        'lotofacil.csv', 
        'gan_generated_lotofacil.csv',
        'selected_numbers.csv'
    )