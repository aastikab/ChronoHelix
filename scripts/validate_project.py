import os
from pathlib import Path
import shutil

def validate_project_structure():
    """Validate and create project directory structure."""
    
    # Define required directories
    required_dirs = [
        'data',
        'data/raw',
        'data/processed',
        'models',
        'utils',
        'scripts',
        'logs',
        'checkpoints',
        'visualizations'
    ]
    
    # Define required __init__.py files
    init_files = [
        'data/__init__.py',
        'models/__init__.py',
        'utils/__init__.py'
    ]
    
    # Create directories
    for dir_path in required_dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)
        print(f"✓ Validated directory: {dir_path}/")
    
    # Create __init__.py files
    for init_file in init_files:
        Path(init_file).touch(exist_ok=True)
        print(f"✓ Validated file: {init_file}")
    
    # Clean up any duplicate/incorrect files
    cleanup_duplicates()
    
    print("\nProject structure validation complete!")

def cleanup_duplicates():
    """Clean up duplicate initialization files."""
    # List of patterns to clean up
    cleanup_patterns = [
        ('data/_init_.py', 'data/__init__.py'),
        ('utils/_init_.py', 'utils/__init__.py'),
        ('models/_init_.py', 'models/__init__.py')
    ]
    
    for old_file, new_file in cleanup_patterns:
        old_path = Path(old_file)
        new_path = Path(new_file)
        
        if old_path.exists():
            if new_path.exists():
                # If both exist, merge content and delete old
                try:
                    old_content = old_path.read_text()
                    new_content = new_path.read_text()
                    merged_content = f"{new_content}\n{old_content}"
                    new_path.write_text(merged_content)
                    old_path.unlink()
                    print(f"✓ Merged and removed duplicate: {old_file}")
                except Exception as e:
                    print(f"! Error merging files: {e}")
            else:
                # If only old exists, rename it
                try:
                    old_path.rename(new_path)
                    print(f"✓ Renamed {old_file} to {new_file}")
                except Exception as e:
                    print(f"! Error renaming file: {e}")

if __name__ == "__main__":
    validate_project_structure() 