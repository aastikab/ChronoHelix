from pathlib import Path
import shutil

def cleanup_init_files():
    """Clean up duplicate initialization files."""
    
    # Directories to check
    dirs = ['models', 'data', 'utils']
    
    for dir_name in dirs:
        dir_path = Path(dir_name)
        old_init = dir_path / '_init_.py'
        new_init = dir_path / '__init__.py'
        
        if old_init.exists():
            if new_init.exists():
                # Merge content if both files exist
                try:
                    old_content = old_init.read_text()
                    new_content = new_init.read_text()
                    merged = f"{new_content}\n\n# Merged from _init_.py\n{old_content}"
                    new_init.write_text(merged)
                    old_init.unlink()
                    print(f"✓ Merged and removed {old_init}")
                except Exception as e:
                    print(f"! Error merging {old_init}: {e}")
            else:
                # Rename if only old file exists
                try:
                    old_init.rename(new_init)
                    print(f"✓ Renamed {old_init} to {new_init}")
                except Exception as e:
                    print(f"! Error renaming {old_init}: {e}")

if __name__ == "__main__":
    cleanup_init_files() 