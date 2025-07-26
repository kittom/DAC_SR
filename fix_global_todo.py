#!/usr/bin/env python3
"""
Fix global TODO files to use correct experiment paths.
This script properly maps each experiment section to its correct paths.
"""

import re
from pathlib import Path

def fix_global_todo_file(todo_file_path):
    """Fix paths in a global TODO file."""
    print(f"Fixing {todo_file_path}")
    
    # Read the file
    with open(todo_file_path, 'r') as f:
        content = f.read()
    
    # Split content into sections by experiment
    sections = re.split(r'(# =+\n# EXPERIMENT \d+: .*\n# Directory: .*\n# =+\n)', content)
    
    fixed_content = ""
    current_experiment_dir = None
    
    for i, section in enumerate(sections):
        if section.startswith('# ='):
            # This is an experiment header
            fixed_content += section
            # Extract experiment directory from the header
            match = re.search(r'# Directory: (.*)', section)
            if match:
                current_experiment_dir = match.group(1)
        else:
            # This is the content section
            if current_experiment_dir and 'Datasets/' in section:
                # Replace relative paths with absolute paths for this experiment
                section = section.replace('"Datasets/', f'"{current_experiment_dir}/Datasets/')
            fixed_content += section
    
    # Write back the fixed content
    with open(todo_file_path, 'w') as f:
        f.write(fixed_content)
    print(f"  Fixed paths in {todo_file_path}")

def main():
    """Fix all global TODO files."""
    print("Fixing global TODO file paths...")
    
    # Fix global TODO files
    global_todo_files = ["global_TODO.txt", "global_TODO_CPU.txt", "global_TODO_GPU.txt"]
    for todo_file in global_todo_files:
        if Path(todo_file).exists():
            fix_global_todo_file(todo_file)
    
    print("Done fixing global TODO file paths!")

if __name__ == "__main__":
    main() 