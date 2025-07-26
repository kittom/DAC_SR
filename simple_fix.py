#!/usr/bin/env python3
"""
Simple script to fix paths in global TODO files.
"""

def fix_todo_file(filename):
    print(f"Fixing {filename}")
    
    with open(filename, 'r') as f:
        content = f.read()
    
    # Replace all relative paths with absolute paths
    # Map each experiment to its correct path
    replacements = {
        'SuiteA_Noise001': 'Experiments/SuiteA/SuiteA_Noise001',
        'SuiteA_Noise005': 'Experiments/SuiteA/SuiteA_Noise005', 
        'SuiteA_Noise01': 'Experiments/SuiteA/SuiteA_Noise01',
        'SuiteA_Noise02': 'Experiments/SuiteA/SuiteA_Noise02',
        'SuiteA_Noise04': 'Experiments/SuiteA/SuiteA_Noise04',
        'SuiteA_NoNoise': 'Experiments/SuiteA/SuiteA_NoNoise'
    }
    
    # First, replace all "Datasets/" with the correct experiment paths
    for exp_name, exp_path in replacements.items():
        # Find the section for this experiment and replace its paths
        start_marker = f"# EXPERIMENT"
        end_marker = "# ============================================================"
        
        # Split content into sections
        sections = content.split("# ============================================================")
        
        for i, section in enumerate(sections):
            if exp_name in section:
                # Replace "Datasets/" with the correct path in this section
                section = section.replace('"Datasets/', f'"{exp_path}/Datasets/')
                sections[i] = section
        
        content = "# ============================================================".join(sections)
    
    with open(filename, 'w') as f:
        f.write(content)
    
    print(f"  Fixed {filename}")

# Fix all global TODO files
files = ["global_TODO_CPU.txt", "global_TODO_GPU.txt", "global_TODO.txt"]
for file in files:
    fix_todo_file(file)

print("Done!") 