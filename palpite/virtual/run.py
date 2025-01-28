#!/usr/bin/env python3
import subprocess
import sys
from datetime import datetime
from pathlib import Path
import os

def get_script_directory():
    """Return the absolute path of the directory containing this script."""
    return os.path.dirname(os.path.abspath(__file__))

def setup_log_file():
    """Setup log file and return its path. Delete existing log if present."""
    script_dir = get_script_directory()
    log_path = os.path.join(script_dir, 'execution.log')
    
    # Remove existing log if present
    if os.path.exists(log_path):
        os.remove(log_path)
        print(f"Removed existing log file: {log_path}")
    
    return log_path

def log_message(log_file, message):
    """Write a timestamped message to the log file."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_file, 'a') as f:
        f.write(f"[{timestamp}] {message}\n")

def run_script(script_name, log_file):
    """Execute a Python script and log its output."""
    script_dir = get_script_directory()
    script_path = os.path.join(script_dir, script_name)
    
    if not os.path.exists(script_path):
        error_msg = f"Script not found: {script_path}"
        log_message(log_file, error_msg)
        return False

    try:
        log_message(log_file, f"Starting execution of {script_name}")
        
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True
        )
        
        # Log stdout if present
        if result.stdout:
            log_message(log_file, "Output:")
            log_message(log_file, result.stdout)
        
        # Log stderr if present
        if result.stderr:
            log_message(log_file, "Errors:")
            log_message(log_file, result.stderr)
        
        # Check return code
        if result.returncode != 0:
            log_message(log_file, f"Script failed with return code {result.returncode}")
            return False
        
        log_message(log_file, f"Successfully executed {script_name}")
        return True
        
    except Exception as e:
        log_message(log_file, f"Error executing {script_name}: {str(e)}")
        return False

def commit_changes(log_file):
    """Commit all updated files with a predefined message."""
    try:
        log_message(log_file, "Starting git commit process")
        
        # Define repository path
        repo_path = "/home/frankalcantara.github.io"
        os.chdir(repo_path)
        
        # Check for changes
        status = subprocess.run(
            ['git', 'status', '--porcelain'],
            capture_output=True,
            text=True
        )
        
        if not status.stdout.strip():
            log_message(log_file, "No changes to commit")
            return True
        
        # Add changes
        subprocess.run(['git', 'add', 'assets/table.html'], check=True)
        subprocess.run(['git', 'add', '.'], check=True)
        log_message(log_file, "Added all changes to git staging")
        
        # Commit
        result = subprocess.run(
            ['git', 'commit', '-m', 'palpites atualizados'],
            check=True,
            capture_output=True,
            text=True
        )
        log_message(log_file, "Commit realizado com sucesso")
        
        return True
        
    except subprocess.CalledProcessError as e:
        log_message(log_file, f"Erro no Git (code {e.returncode}): {e.stderr}")
        return False
    except Exception as e:
        log_message(log_file, f"Erro inesperado: {str(e)}")
        return False

def main():
    """Main execution function that runs scripts in sequence and commits changes."""
    # Setup log file
    log_file = setup_log_file()
    log_message(log_file, "Starting sequential script execution")
    
    # List of scripts to execute in order
    scripts = [
        "downloadSorteios.py",
        "novoCheca.py",
        "GAN.py",
        "removedup.py",
        "send2.py"
    ]
    
    # Log execution environment
    log_message(log_file, f"Current working directory: {os.getcwd()}")
    log_message(log_file, f"Script directory: {get_script_directory()}")
    
    # Execute each script
    for script in scripts:
        if not run_script(script, log_file):
            log_message(log_file, f"Execution chain stopped at {script}")
            sys.exit(1)
    
    # After all scripts execute successfully, commit changes
    if commit_changes(log_file):
        log_message(log_file, "All scripts executed and changes committed successfully")
    else:
        log_message(log_file, "Scripts executed but git commit failed")
        sys.exit(1)

if __name__ == "__main__":
    main()