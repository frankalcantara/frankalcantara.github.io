import requests
import pandas as pd
import io
from datetime import datetime
import os
from pathlib import Path

def check_and_download_lotofacil():
    url = "https://jolly-flower-cfbe.frank-alcantara.workers.dev/"
    last_update_file = 'last_update_info.txt'
    output_csv = 'lotofacil.csv'
    
    def get_last_saved_date():
        try:
            if os.path.exists(output_csv):
                df = pd.read_csv(output_csv)
                # Convert the date column to datetime, assuming it's the first column
                last_date = pd.to_datetime(df.iloc[-1, 0])
                return last_date
            return None
        except Exception as e:
            print(f"Error reading last saved date: {e}")
            return None
    
    try:
        # Get last saved date
        last_saved_date = get_last_saved_date()
        
        # Download current file to check
        response = requests.get(url)
        response.raise_for_status()
        
        # Read Excel file from response
        current_df = pd.read_excel(
            io.BytesIO(response.content),
            usecols="B:Q",
            skiprows=1
        )
        
        # Get the last date from current file
        current_last_date = pd.to_datetime(current_df.iloc[-1, 0])
        
        if last_saved_date is None or current_last_date > last_saved_date:
            # Save new file
            current_df.to_csv(output_csv, index=False)
            print(f"New data found! File updated. Last date: {current_last_date.strftime('%Y-%m-%d')}")
            return True
        else:
            print(f"No new data available. Last saved date: {last_saved_date.strftime('%Y-%m-%d')}")
            return False
            
    except requests.RequestException as e:
        print(f"Error downloading the file: {e}")
        return False
    except Exception as e:
        print(f"Error processing the file: {e}")
        return False

if __name__ == "__main__":
    #apagando o arquivo antes de baixar um novo
    os.remove('lotofacil.csv') if os.path.exists('lotofacil.csv') else None
    
    check_and_download_lotofacil()