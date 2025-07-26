#!/usr/bin/env python3
"""
Master Experiment Runner

This script orchestrates multiple experiments by:
1. Reading multiple config files from the Configs/ directory
2. Setting up each experiment (directories, logging, etc.)
3. Generating data for all experiments
4. Creating individual TODO.txt files for each experiment
5. Consolidating all TODO lists into a global TODO.txt

Usage:
    python master_experiment_runner.py [--config-dir Configs] [--global-todo global_TODO.txt]
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any

# Import the single experiment runner
from experiment_runner import ExperimentRunner


class MasterExperimentRunner:
    def __init__(self, config_dir: str = "Configs", global_todo_file: str = "global_TODO.txt"):
        self.config_dir = Path(config_dir)
        self.global_todo_file = Path(global_todo_file)
        self.global_gpu_todo_file = Path("global_TODO_GPU.txt")
        self.global_cpu_todo_file = Path("global_TODO_CPU.txt")
        self.experiment_runners = []
        self.setup_logging()
        
    def setup_logging(self):
        """Set up logging for the master runner."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('master_experiment_runner.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def find_config_files(self) -> List[Path]:
        """Find all JSON config files in the config directory."""
        config_files = list(self.config_dir.glob("*.json"))
        self.logger.info(f"Found {len(config_files)} config files: {[f.name for f in config_files]}")
        return config_files
        
    def load_config(self, config_file: Path) -> Dict[str, Any]:
        """Load a single config file."""
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            self.logger.info(f"Loaded config: {config_file.name}")
            return config
        except Exception as e:
            self.logger.error(f"Failed to load config {config_file}: {e}")
            raise
            
    def setup_experiments(self, config_files: List[Path]) -> List[ExperimentRunner]:
        """Set up all experiments from config files."""
        experiment_runners = []
        
        for config_file in config_files:
            try:
                config = self.load_config(config_file)
                experiment_name = config.get('experiment', {}).get('name', config_file.stem)
                
                self.logger.info(f"Setting up experiment: {experiment_name}")
                
                # Create experiment runner with config file path
                runner = ExperimentRunner(str(config_file))
                
                # Setup phase is already done in __init__ (creates directories)
                # Just verify the experiment directory was created
                if not runner.experiment_dir.exists():
                    raise RuntimeError(f"Experiment directory was not created: {runner.experiment_dir}")
                
                experiment_runners.append(runner)
                self.logger.info(f"Successfully set up experiment: {experiment_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to set up experiment from {config_file}: {e}")
                continue
                
        return experiment_runners
        
    def generate_data_for_all(self, experiment_runners: List[ExperimentRunner]):
        """Generate data for all experiments."""
        self.logger.info("Generating data for all experiments...")
        
        for runner in experiment_runners:
            try:
                experiment_name = runner.config.get('experiment', {}).get('name', 'Unknown')
                self.logger.info(f"Generating data for experiment: {experiment_name}")
                runner.generate_data()
            except Exception as e:
                self.logger.error(f"Failed to generate data for experiment {experiment_name}: {e}")
                continue
                
    def create_todo_for_all(self, experiment_runners: List[ExperimentRunner]):
        """Create TODO.txt files for all experiments."""
        self.logger.info("Creating TODO.txt files for all experiments...")
        
        for runner in experiment_runners:
            try:
                experiment_name = runner.config.get('experiment', {}).get('name', 'Unknown')
                self.logger.info(f"Creating TODO.txt for experiment: {experiment_name}")
                runner.run_evaluations()
            except Exception as e:
                self.logger.error(f"Failed to create TODO.txt for experiment {experiment_name}: {e}")
                continue
                
    def consolidate_todo_lists(self, experiment_runners: List[ExperimentRunner]):
        """Consolidate all individual TODO.txt files into separate global TODO files for GPU and CPU."""
        self.logger.info(f"Consolidating TODO lists into separate GPU and CPU files...")
        
        # Initialize global TODO files
        with open(self.global_gpu_todo_file, 'w') as gpu_todo:
            gpu_todo.write("# Global TODO List - GPU Algorithms Only\n")
            gpu_todo.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            gpu_todo.write(f"# Total experiments: {len(experiment_runners)}\n")
            gpu_todo.write("# Run on GPU machine\n\n")
            
        with open(self.global_cpu_todo_file, 'w') as cpu_todo:
            cpu_todo.write("# Global TODO List - CPU Algorithms Only\n")
            cpu_todo.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            cpu_todo.write(f"# Total experiments: {len(experiment_runners)}\n")
            cpu_todo.write("# Run on CPU machine\n\n")
            
        # Also create the combined file for backward compatibility
        with open(self.global_todo_file, 'w') as combined_todo:
            combined_todo.write("# Global TODO List - All Algorithms (Combined)\n")
            combined_todo.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            combined_todo.write(f"# Total experiments: {len(experiment_runners)}\n")
            combined_todo.write("# Note: Use global_TODO_GPU.txt and global_TODO_CPU.txt for separate execution\n\n")
            
        for i, runner in enumerate(experiment_runners, 1):
            experiment_name = runner.config.get('experiment', {}).get('name', 'Unknown')
            experiment_gpu_todo_file = runner.experiment_dir / "TODO_GPU.txt"
            experiment_cpu_todo_file = runner.experiment_dir / "TODO_CPU.txt"
            experiment_todo_file = runner.experiment_dir / "TODO.txt"
            
            self.logger.info(f"Processing TODO files from experiment {i}: {experiment_name}")
            
            # Process GPU TODO file
            if experiment_gpu_todo_file.exists():
                self.logger.info(f"Adding GPU TODO list from experiment: {experiment_name}")
                with open(self.global_gpu_todo_file, 'a') as gpu_todo:
                    gpu_todo.write(f"# {'='*60}\n")
                    gpu_todo.write(f"# EXPERIMENT {i}: {experiment_name}\n")
                    gpu_todo.write(f"# Directory: {runner.experiment_dir}\n")
                    gpu_todo.write(f"# {'='*60}\n\n")
                    
                    with open(experiment_gpu_todo_file, 'r') as exp_gpu_todo:
                        content = exp_gpu_todo.read()
                        # Convert relative paths to absolute paths
                        content = content.replace('"Datasets/', f'"{runner.experiment_dir}/Datasets/')
                        gpu_todo.write(content)
                        gpu_todo.write("\n\n")
            else:
                self.logger.warning(f"TODO_GPU.txt not found for experiment: {experiment_name}")
            
            # Process CPU TODO file
            if experiment_cpu_todo_file.exists():
                self.logger.info(f"Adding CPU TODO list from experiment: {experiment_name}")
                with open(self.global_cpu_todo_file, 'a') as cpu_todo:
                    cpu_todo.write(f"# {'='*60}\n")
                    cpu_todo.write(f"# EXPERIMENT {i}: {experiment_name}\n")
                    cpu_todo.write(f"# Directory: {runner.experiment_dir}\n")
                    cpu_todo.write(f"# {'='*60}\n\n")
                    
                    with open(experiment_cpu_todo_file, 'r') as exp_cpu_todo:
                        content = exp_cpu_todo.read()
                        # Convert relative paths to absolute paths
                        content = content.replace('"Datasets/', f'"{runner.experiment_dir}/Datasets/')
                        cpu_todo.write(content)
                        cpu_todo.write("\n\n")
            else:
                self.logger.warning(f"TODO_CPU.txt not found for experiment: {experiment_name}")
            
            # Process combined TODO file for backward compatibility
            if experiment_todo_file.exists():
                with open(self.global_todo_file, 'a') as combined_todo:
                    combined_todo.write(f"# {'='*60}\n")
                    combined_todo.write(f"# EXPERIMENT {i}: {experiment_name}\n")
                    combined_todo.write(f"# Directory: {runner.experiment_dir}\n")
                    combined_todo.write(f"# {'='*60}\n\n")
                    
                    with open(experiment_todo_file, 'r') as exp_todo:
                        content = exp_todo.read()
                        # Convert relative paths to absolute paths
                        content = content.replace('"Datasets/', f'"{runner.experiment_dir}/Datasets/')
                        combined_todo.write(content)
                        combined_todo.write("\n\n")
            else:
                self.logger.warning(f"TODO.txt not found for experiment: {experiment_name}")
                    
        self.logger.info(f"Global TODO files created:")
        self.logger.info(f"  - GPU algorithms: {self.global_gpu_todo_file}")
        self.logger.info(f"  - CPU algorithms: {self.global_cpu_todo_file}")
        self.logger.info(f"  - Combined: {self.global_todo_file}")
        
    def run_all_experiments(self):
        """Run the complete master experiment pipeline."""
        self.logger.info("Starting master experiment runner...")
        
        # Find all config files
        config_files = self.find_config_files()
        if not config_files:
            self.logger.error(f"No config files found in {self.config_dir}")
            return
            
        # Set up all experiments
        self.logger.info("Phase 1: Setting up experiments...")
        experiment_runners = self.setup_experiments(config_files)
        
        if not experiment_runners:
            self.logger.error("No experiments were successfully set up")
            return
            
        # Generate data for all experiments
        self.logger.info("Phase 2: Generating data for all experiments...")
        self.generate_data_for_all(experiment_runners)
        
        # Create TODO files for all experiments
        self.logger.info("Phase 3: Creating TODO files for all experiments...")
        self.create_todo_for_all(experiment_runners)
        
        # Consolidate all TODO lists
        self.logger.info("Phase 4: Consolidating TODO lists...")
        self.consolidate_todo_lists(experiment_runners)
        
        self.logger.info("Master experiment runner completed successfully!")
        self.logger.info(f"Global TODO files available:")
        self.logger.info(f"  - GPU algorithms: {self.global_gpu_todo_file}")
        self.logger.info(f"  - CPU algorithms: {self.global_cpu_todo_file}")
        self.logger.info(f"  - Combined: {self.global_todo_file}")
        self.logger.info("Run './run_global.sh -t global_TODO_GPU.txt -j 16' on GPU machine")
        self.logger.info("Run './run_global.sh -t global_TODO_CPU.txt -j 16' on CPU machine")


def main():
    parser = argparse.ArgumentParser(description="Master Experiment Runner")
    parser.add_argument("--config-dir", default="Configs", 
                       help="Directory containing config files (default: Configs)")
    parser.add_argument("--global-todo", default="global_TODO.txt",
                       help="Output file for consolidated TODO list (default: global_TODO.txt)")
    
    args = parser.parse_args()
    
    # Create master runner
    master_runner = MasterExperimentRunner(
        config_dir=args.config_dir,
        global_todo_file=args.global_todo
    )
    
    # Run all experiments
    master_runner.run_all_experiments()


if __name__ == "__main__":
    main() 