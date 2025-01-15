#!/usr/bin/env python3
import subprocess
import sys
import logging
from datetime import datetime
from typing import List, Tuple
import glob
import os

def setup_logging() -> logging.Logger:
    """Configure logging with timestamp and appropriate format.
    
    Returns:
        logging.Logger: Configured logger instance with file and console handlers
    """
    # Ensuring the log is created in the same directory as the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(
        script_dir,
        f'script_execution_{datetime.now():%Y%m%d_%H%M%S}.log'
    )
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_script(script_name: str, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Execute a Python script and return its success status and output.
    
    Args:
        script_name: Name of the Python script to execute
        logger: Logger instance for recording execution details
    
    Returns:
        Tuple containing execution success (bool) and output/error message (str)
    """
    # Getting absolute path of the current script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, script_name)
    
    # Verifying file existence before execution
    if not os.path.exists(script_path):
        error_msg = f"Script not found: {script_path}"
        logger.error(error_msg)
        return False, error_msg

    try:
        logger.info(f"Starting execution of {script_name}")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully executed {script_name}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error in {script_name}: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error running {script_name}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def commit_changes(logger: logging.Logger) -> bool:
    """
    Commit all updated files with a predefined message.
    
    Args:
        logger: Logger instance for recording git operations
        
    Returns:
        bool: True if commit was successful, False otherwise
    """
    try:
        logger.info("Starting git commit process")
        
        # Add all changes to staging
        subprocess.run(['git', 'add', '/home/frankalcantara.github.io/assets/table.html'], check=True)
        
        subprocess.run(['git', 'add', '.'], check=True)
        logger.info("Added all changes to git staging")
        
        # Create commit with message
        subprocess.run(['git', 'commit', '-m', 'palpites atualizados'], check=True)
        logger.info("Successfully created commit with message: 'palpites atualizados'")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during git operations: {str(e)}")
        return False

def main():
    """
    Main execution function that runs scripts in sequence and commits changes.
    """

    # Apaga todos os arquivos .log no diretório atual e subdiretorios
    [os.remove(f) for f in glob.glob("**/*.log", recursive=True)]

    logger = setup_logging()
    
    # List of scripts to execute in order
    scripts = [
        "downloadLotoFacil.py",
        "novoCheca.py",
        "GAN.py",
        "todb.py",
        "send2.py"
    ]
    
    logger.info("Starting sequential script execution")
    
    # Logging the execution environment for debugging
    logger.info(f"Current working directory: {os.getcwd()}")
    logger.info(f"Script location: {os.path.dirname(os.path.abspath(__file__))}")
    
    for script in scripts:
        success, output = run_script(script, logger)
        if not success:
            logger.error(f"Execution chain stopped at {script}")
            sys.exit(1)
        
        logger.info(f"Output from {script}:\n{output}")
    
    # After all scripts execute successfully, commit changes
    if commit_changes(logger):
        logger.info("All scripts executed and changes committed successfully")
    else:
        logger.error("Scripts executed but git commit failed")
        sys.exit(1)

if __name__ == "__main__":
    main()