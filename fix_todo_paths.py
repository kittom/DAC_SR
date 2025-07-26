#!/usr/bin/env python3
"""
Fix TODO file paths to use absolute paths instead of relative paths.
This script updates all TODO files to use the correct experiment directory paths.
"""

import os
import glob
from pathlib import Path

def fix_todo_file(todo_file_path):
    """Fix paths in a single TODO file."""
    print(f"Fixing {todo_file_path}")
    
    # Read the file
    with open(todo_file_path, 'r') as f:
        content = f.read()
    
    # Find the experiment directory by looking at the TODO file path
    todo_path = Path(todo_file_path)
    if 'Experiments' in str(todo_path):
        # Extract experiment directory from TODO file path
        parts = todo_path.parts
        experiments_index = parts.index('Experiments')
        experiment_dir = Path(*parts[experiments_index:-1])  # Everything from Experiments to parent of TODO file
        
        # Replace relative paths with absolute paths
        old_content = content
        content = content.replace('"Datasets/', f'"{experiment_dir}/Datasets/')
        
        if old_content != content:
            # Write back the fixed content
            with open(todo_file_path, 'w') as f:
                f.write(content)
            print(f"  Fixed paths in {todo_file_path}")
        else:
            print(f"  No changes needed for {todo_file_path}")
    else:
        # This is a global TODO file - need to fix all experiment paths
        print(f"  Fixing global TODO file {todo_file_path}")
        
        # Find all experiment directories
        experiment_dirs = []
        for exp_dir in Path("Experiments").glob("**/SuiteA/*"):
            if exp_dir.is_dir() and any(exp_dir.glob("Datasets/*")):
                experiment_dirs.append(exp_dir)
        
        # Replace each experiment's relative paths with absolute paths
        old_content = content
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            # Replace paths for this specific experiment
            content = content.replace(f'"Datasets/', f'"{exp_dir}/Datasets/')
        
        if old_content != content:
            # Write back the fixed content
            with open(todo_file_path, 'w') as f:
                f.write(content)
            print(f"  Fixed paths in {todo_file_path}")
        else:
            print(f"  No changes needed for {todo_file_path}")

def main():
    """Fix all TODO files."""
    print("Fixing TODO file paths...")
    
    # Fix individual experiment TODO files
    todo_files = glob.glob("Experiments/**/TODO*.txt", recursive=True)
    for todo_file in todo_files:
        fix_todo_file(todo_file)
    
    # Fix global TODO files
    global_todo_files = ["global_TODO.txt", "global_TODO_CPU.txt", "global_TODO_GPU.txt"]
    for todo_file in global_todo_files:
        if os.path.exists(todo_file):
            fix_todo_file(todo_file)
    
    print("Done fixing TODO file paths!")

if __name__ == "__main__":
    main() 