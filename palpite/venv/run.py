#!/usr/bin/env python3
import subprocess
import sys
import logging
from datetime import datetime
from typing import List, Tuple

def setup_logging() -> logging.Logger:
    """Configure logging with timestamp and appropriate format."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'script_execution_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    return logging.getLogger(__name__)

def run_script(script_path: str, logger: logging.Logger) -> Tuple[bool, str]:
    """
    Execute a Python script and return its success status and output.
    
    Args:
        script_path: Path to the Python script to execute
        logger: Logger instance for recording execution details
    
    Returns:
        Tuple containing execution success (bool) and output/error message (str)
    """
    try:
        logger.info(f"Starting execution of {script_path}")
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            check=True
        )
        logger.info(f"Successfully executed {script_path}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        error_msg = f"Error in {script_path}: {e.stderr}"
        logger.error(error_msg)
        return False, error_msg
    except Exception as e:
        error_msg = f"Unexpected error running {script_path}: {str(e)}"
        logger.error(error_msg)
        return False, error_msg

def main():
    logger = setup_logging()
    
    # List of scripts to execute in order
    scripts = [
        "downloadLotoFacil.py",
        "chegaResultado.py",
        "GAN.py",
        "todb.py",
        "send2.py"
    ]
    
    logger.info("Starting sequential script execution")
    
    for script in scripts:
        success, output = run_script(script, logger)
        if not success:
            logger.error(f"Execution chain stopped at {script}")
            sys.exit(1)
        
        logger.info(f"Output from {script}:\n{output}")
    
    logger.info("All scripts executed successfully")

if __name__ == "__main__":
    main()