#!/usr/bin/env python3
"""
Experiment Runner for Symbolic Regression Experiments

This script reads a config.json file and orchestrates the entire experiment pipeline:
1. Data generation based on benchmark specifications
2. Evaluation using specified SR algorithms
3. Result collection and reporting
"""

import json
import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional
import shutil

class ExperimentRunner:
    def __init__(self, config_path: str):
        """Initialize the experiment runner with a config file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.setup_logging()
        
        # Set up paths
        self.project_root = Path(__file__).parent
        self.experiment_dir = self._setup_experiment_directory()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load and validate the configuration file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
            
        with open(self.config_path, 'r') as f:
            config = json.load(f)
            
        # Basic validation
        required_sections = ['experiment', 'data_generation', 'evaluation', 'output']
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required section: {section}")
                
        return config
    
    def setup_logging(self):
        """Set up logging configuration."""
        log_level = getattr(logging, self.config['execution']['logging']['level'])
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(self.config['execution']['logging']['log_file'])
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
    def _setup_experiment_directory(self) -> Path:
        """Set up the experiment directory structure."""
        output_config = self.config['output']
        base_dir = Path(output_config['base_directory'])
        suite_dir = base_dir / output_config['suite_name']
        experiment_dir = suite_dir / output_config['experiment_name']
        
        # Create directories
        experiment_dir.mkdir(parents=True, exist_ok=True)
        (experiment_dir / 'Datasets').mkdir(exist_ok=True)
        (experiment_dir / 'Results').mkdir(exist_ok=True)
        (experiment_dir / 'Logs').mkdir(exist_ok=True)
        
        # Copy config to experiment directory
        config_copy_path = experiment_dir / 'config.json'
        shutil.copy2(self.config_path, config_copy_path)
        
        return experiment_dir
    
    def generate_data(self):
        """Generate datasets based on configuration."""
        self.logger.info("Starting data generation...")
        
        data_config = self.config['data_generation']
        datasets_dir = self.experiment_dir / 'Datasets'
        
        # Generate data for each enabled benchmark
        for benchmark_name, benchmark_config in data_config['benchmarks'].items():
            if not benchmark_config['enabled']:
                continue
                
            self.logger.info(f"Generating data for {benchmark_name}...")
            
            if benchmark_name in ['leadingones', 'onemax']:
                self._generate_discrete_benchmark_data(benchmark_name, benchmark_config, datasets_dir)
            elif benchmark_name == 'psacmaes':
                self._generate_psacmaes_data(benchmark_config, datasets_dir)
            elif benchmark_name == 'model':
                self._generate_model_data(benchmark_config, datasets_dir)
    
    def _generate_discrete_benchmark_data(self, benchmark_name: str, config: Dict, datasets_dir: Path):
        """Generate data for LeadingOnes and OneMax benchmarks."""
        data_type = config['data_type']
        instance_sizes = config['instance_sizes']
        
        # Create benchmark directory (no longer need data_type subdirectory)
        benchmark_dir = datasets_dir / benchmark_name.capitalize()
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Determine the correct generator script
        if benchmark_name == 'leadingones':
            generator_script = self.project_root / 'DataGeneration' / 'Generators' / 'LeadingOnes' / 'LeadingOnesGT.py'
        elif benchmark_name == 'onemax':
            generator_script = self.project_root / 'DataGeneration' / 'Generators' / 'OneMax' / 'OneMaxGT.py'
        else:
            raise ValueError(f"Unknown benchmark: {benchmark_name}")
        
        # Convert instance sizes to string arguments
        size_args = [str(size) for size in instance_sizes]
        
        # Determine which evaluation types are enabled
        eval_config = self.config['evaluation']['analysis_styles']
        enabled_evaluations = []
        if eval_config['control']['enabled']:
            enabled_evaluations.append('control')
        if eval_config['library']['enabled']:
            enabled_evaluations.append('library')
        if eval_config['rounding']['enabled']:
            enabled_evaluations.append('rounding')
        
        # Run the generator with conda environment activation
        conda_env = 'generation'  # Use the generation environment for data generation
        evaluation_type_arg = f"--evaluation-type {enabled_evaluations[0]}" if len(enabled_evaluations) == 1 else ""
        cmd = [
            'bash', '-c',
            f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && python3 {generator_script} {" ".join(size_args)} --data-type {data_type} --output-dir {benchmark_dir} {evaluation_type_arg}'
        ]
        
        self.logger.info(f"Running data generation for {benchmark_name} in {conda_env} environment")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"Data generation failed for {benchmark_name}: {result.stderr}")
            raise RuntimeError(f"Data generation failed for {benchmark_name}")
        else:
            self.logger.info(f"Successfully generated {benchmark_name} data")
    
    def _generate_psacmaes_data(self, config: Dict, datasets_dir: Path):
        """Generate PSA-CMA-ES benchmark data."""
        data_type = config['data_type']
        iterations = config['iterations']
        
        # Create benchmark directory (no longer need data_type subdirectory)
        benchmark_dir = datasets_dir / 'PSACMAES'
        benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        generator_script = self.project_root / 'DataGeneration' / 'Generators' / 'PSA_CMA_ES' / 'generate_ground_truth.py'
        
        # Run the generator with conda environment activation
        conda_env = 'generation'  # Use the generation environment for data generation
        cmd = [
            'bash', '-c',
            f'source ~/miniconda3/etc/profile.d/conda.sh && conda activate {conda_env} && python3 {generator_script} --iterations {iterations} --data-type {data_type} --output-root {benchmark_dir}'
        ]
        
        self.logger.info(f"Running PSA-CMA-ES data generation in {conda_env} environment")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            self.logger.error(f"PSA-CMA-ES data generation failed: {result.stderr}")
            raise RuntimeError("PSA-CMA-ES data generation failed")
        else:
            self.logger.info("Successfully generated PSA-CMA-ES data")
    
    def _generate_model_data(self, config: Dict, datasets_dir: Path):
        """Generate model-specific data."""
        # This would be implemented based on specific model requirements
        self.logger.warning("Model data generation not yet implemented")
    

    
    def run_evaluations(self):
        """Run SR algorithm evaluations based on configuration."""
        self.logger.info("Starting SR algorithm evaluations...")
        
        eval_config = self.config['evaluation']
        datasets_dir = self.experiment_dir / 'Datasets'
        results_dir = self.experiment_dir / 'Results'
        
        # Run each enabled analysis style
        for style_name, style_config in eval_config['analysis_styles'].items():
            if not style_config['enabled']:
                continue
                
            self.logger.info(f"Running {style_name} evaluation...")
            self._run_analysis_style(style_name, datasets_dir, results_dir)
    
    def _run_analysis_style(self, style_name: str, datasets_dir: Path, results_dir: Path):
        """Run a specific analysis style (control, library, rounding)."""
        utils_dir = self.project_root / 'Experiments' / 'script_utils' / 'evaluation'
        
        # Get noise level from config
        noise_level = self.config['data_generation']['noise']['level']
        
        # Process each benchmark's CSV files
        for benchmark_name, benchmark_config in self.config['data_generation']['benchmarks'].items():
            if not benchmark_config['enabled']:
                continue
                
            data_type = benchmark_config['data_type']
            
            if benchmark_name in ['leadingones', 'onemax']:
                # Handle LeadingOnes and OneMax
                if benchmark_name == 'leadingones':
                    csv_filename = "GTLeadingOnes.csv"  # Correct capitalization
                elif benchmark_name == 'onemax':
                    csv_filename = "GTOneMax.csv"  # Correct capitalization
                csv_path = datasets_dir / benchmark_name.capitalize() / csv_filename
                
                if csv_path.exists():
                    self.logger.info(f"Processing {benchmark_name} {data_type} data: {csv_path}")
                    
                    if style_name == 'control':
                        script_path = utils_dir / 'evaluate_control_library.sh'
                        cmd = ['bash', str(script_path), str(csv_path), str(noise_level)]
                    elif style_name == 'library':
                        script_path = utils_dir / 'evaluate_tailored_library.sh'
                        problem_type = 'leading_ones' if benchmark_name == 'leadingones' else 'one_max'
                        cmd = ['bash', str(script_path), problem_type, str(csv_path), str(noise_level)]
                    elif style_name == 'rounding':
                        script_path = utils_dir / 'evaluate_rounding.sh'
                        cmd = ['bash', str(script_path), str(csv_path), str(noise_level)]
                    else:
                        raise ValueError(f"Unknown analysis style: {style_name}")
                    
                    self.logger.info(f"Running: {' '.join(cmd)}")
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    if result.returncode != 0:
                        self.logger.error(f"{style_name} evaluation failed for {benchmark_name}: {result.stderr}")
                        raise RuntimeError(f"{style_name} evaluation failed for {benchmark_name}")
                    else:
                        self.logger.info(f"Successfully completed {style_name} evaluation for {benchmark_name}")
                else:
                    self.logger.warning(f"CSV file not found: {csv_path}")
            
            elif benchmark_name == 'psacmaes':
                # Handle PSA-CMA-ES benchmarks
                for sub_benchmark in benchmark_config['benchmarks']:
                    # PSA-CMA-ES creates subdirectories with lowercase names
                    sub_benchmark_lower = sub_benchmark.lower().replace('_', '')
                    csv_filename = "psa_vars.csv"  # The actual data file
                    csv_path = datasets_dir / 'PSACMAES' / sub_benchmark_lower / csv_filename
                    
                    if csv_path.exists():
                        self.logger.info(f"Processing PSA-CMA-ES {sub_benchmark} {data_type} data: {csv_path}")
                        
                        if style_name == 'control':
                            script_path = utils_dir / 'evaluate_control_library.sh'
                            cmd = ['bash', str(script_path), str(csv_path), str(noise_level)]
                        elif style_name == 'library':
                            script_path = utils_dir / 'evaluate_tailored_library.sh'
                            cmd = ['bash', str(script_path), 'psacmaes', str(csv_path), str(noise_level)]
                        elif style_name == 'rounding':
                            script_path = utils_dir / 'evaluate_rounding.sh'
                            cmd = ['bash', str(script_path), str(csv_path), str(noise_level)]
                        else:
                            raise ValueError(f"Unknown analysis style: {style_name}")
                        
                        self.logger.info(f"Running: {' '.join(cmd)}")
                        result = subprocess.run(cmd, capture_output=True, text=True)
                        
                        if result.returncode != 0:
                            self.logger.error(f"{style_name} evaluation failed for PSA-CMA-ES {sub_benchmark}: {result.stderr}")
                            raise RuntimeError(f"{style_name} evaluation failed for PSA-CMA-ES {sub_benchmark}")
                        else:
                            self.logger.info(f"Successfully completed {style_name} evaluation for PSA-CMA-ES {sub_benchmark}")
                    else:
                        self.logger.warning(f"CSV file not found: {csv_path}")
    
    def collect_results(self):
        """Collect and organize results."""
        self.logger.info("Collecting results...")
        
        # This would implement result collection logic
        # For now, just log that it's complete
        self.logger.info("Results collection completed")
    
    def generate_report(self):
        """Generate a summary report of the experiment."""
        self.logger.info("Generating experiment report...")
        
        report_path = self.experiment_dir / 'experiment_report.txt'
        
        with open(report_path, 'w') as f:
            f.write(f"Experiment Report\n")
            f.write(f"=================\n\n")
            f.write(f"Experiment: {self.config['experiment']['name']}\n")
            f.write(f"Description: {self.config['experiment']['description']}\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"- Config file: {self.config_path}\n")
            f.write(f"- Experiment directory: {self.experiment_dir}\n\n")
            
            f.write(f"Data Generation:\n")
            for benchmark, config in self.config['data_generation']['benchmarks'].items():
                if config['enabled']:
                    f.write(f"- {benchmark}: {config['data_type']} data\n")
            f.write(f"- Noise: {self.config['data_generation']['noise']['enabled']}\n")
            f.write(f"- Dropout: {self.config['data_generation']['dropout']['enabled']}\n\n")
            
            f.write(f"Evaluation:\n")
            for style, config in self.config['evaluation']['analysis_styles'].items():
                if config['enabled']:
                    f.write(f"- {style}: {config['description']}\n")
        
        self.logger.info(f"Report generated: {report_path}")
    
    def run_experiment(self):
        """Run the complete experiment pipeline."""
        self.logger.info("Starting experiment pipeline...")
        
        try:
            # Step 1: Generate data
            self.generate_data()
            
            # Step 2: Run evaluations
            self.run_evaluations()
            
            # Step 3: Collect results
            self.collect_results()
            
            # Step 4: Generate report
            self.generate_report()
            
            self.logger.info("Experiment completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Experiment failed: {e}")
            raise

def main():
    parser = argparse.ArgumentParser(description='Run symbolic regression experiments from config')
    parser.add_argument('config', help='Path to config.json file')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be run without executing')
    parser.add_argument('--phase', choices=['setup', 'generate', 'evaluate', 'analyze', 'all'], default='all',
                       help='Which phase of the experiment to run (default: all)')
    
    args = parser.parse_args()
    
    try:
        runner = ExperimentRunner(args.config)
        
        if args.dry_run:
            print("Dry run mode - would execute:")
            print(f"Config: {args.config}")
            print(f"Experiment directory: {runner.experiment_dir}")
            print(f"Phase: {args.phase}")
            print("Data generation and evaluation would be run")
        else:
            if args.phase == 'setup':
                print("Setup phase completed - experiment directory created")
            elif args.phase == 'generate':
                runner.generate_data()
                print("Data generation completed")
            elif args.phase == 'evaluate':
                runner.run_evaluations()
                print("Evaluation completed")
            elif args.phase == 'analyze':
                runner.collect_results()
                runner.generate_report()
                print("Analysis completed")
            elif args.phase == 'all':
                runner.run_experiment()
            else:
                print(f"Unknown phase: {args.phase}")
                sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main() 