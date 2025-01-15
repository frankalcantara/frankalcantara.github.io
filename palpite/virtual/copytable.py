#!/usr/bin/env python3
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime
import logging

def setup_logging():
    """Configure logging to track operations and errors."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'file_operations_{datetime.now():%Y%m%d_%H%M%S}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def ensure_git_directory(dir_path: str, logger: logging.Logger) -> bool:
    """
    Ensure the directory is tracked by Git by creating a .gitkeep file if needed.
    
    Args:
        dir_path: Path to the directory that should be tracked
        logger: Logger instance for operation tracking
    
    Returns:
        bool: True if directory is now tracked, False if operation failed
    """
    try:
        # Create directory if it doesn't exist
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        
        # Create .gitkeep file to ensure directory is tracked
        gitkeep_path = os.path.join(dir_path, '.gitkeep')
        if not os.path.exists(gitkeep_path):
            Path(gitkeep_path).touch()
            logger.info(f"Created .gitkeep in {dir_path}")
        
        # Add directory to git
        subprocess.run(['git', 'add', dir_path], check=True)
        logger.info(f"Added directory {dir_path} to git")
        
        # Create initial commit for directory if needed
        try:
            subprocess.run(['git', 'commit', '-m', f"Adding Docs directory for tracking"], check=True)
            logger.info("Created commit for Docs directory")
        except subprocess.CalledProcessError:
            # If there's nothing to commit, that's okay
            logger.info("Directory already tracked in git")
        
        return True
    except Exception as e:
        logger.error(f"Error ensuring git directory: {str(e)}")
        return False

def copy_table_file(source_path: str, dest_dir: str, logger: logging.Logger) -> bool:
    """
    Copy table.html from source to destination directory.
    
    Args:
        source_path: Path to source table.html
        dest_dir: Destination directory path
        logger: Logger instance for operation tracking
    
    Returns:
        bool: True if copy successful, False otherwise
    """
    try:
        shutil.copy2(source_path, dest_dir)
        logger.info(f"Successfully copied table.html to {dest_dir}")
        return True
    except Exception as e:
        logger.error(f"Error copying file: {str(e)}")
        return False

def git_operations(file_path: str, logger: logging.Logger) -> bool:
    """
    Add file to git and create commit.
    
    Args:
        file_path: Path to the file to be committed
        logger: Logger instance for operation tracking
    
    Returns:
        bool: True if git operations successful, False otherwise
    """
    try:
        # Add file to git
        subprocess.run(['git', 'add', file_path], check=True)
        logger.info(f"Added {file_path} to git staging area")
        
        # Create commit
        commit_msg = "atualizando tabela"
        subprocess.run(['git', 'commit', '-m', commit_msg], check=True)
        logger.info("Created commit with message: 'atualizando tabela'")
        
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Git operation failed: {str(e)}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error during git operations: {str(e)}")
        return False

def main():
    # Initialize logger
    logger = setup_logging()
    
    # Define paths
    source_path = '/var/www/html/table.html'
    current_dir = os.path.dirname(os.path.abspath(__file__))
    docs_dir = os.path.join(current_dir, 'docs')
    dest_file_path = os.path.join(docs_dir, 'table.html')
    
    logger.info("Starting file copy and git operations")
    
    # First ensure the Docs directory is tracked in git
    if not ensure_git_directory(docs_dir, logger):
        logger.error("Failed to ensure git directory tracking. Exiting.")
        return
    
    # Copy file
    if not copy_table_file(source_path, docs_dir, logger):
        logger.error("File copy operation failed. Exiting.")
        return
    
    # Perform git operations
    if not git_operations(dest_file_path, logger):
        logger.error("Git operations failed. Exiting.")
        return
    
    logger.info("All operations completed successfully")

if __name__ == "__main__":
    main()