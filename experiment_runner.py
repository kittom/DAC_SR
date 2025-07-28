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
        # Use default logging configuration since 'execution' section may not exist
        log_level = logging.INFO
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        logging.basicConfig(
            level=log_level,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler('experiment.log')
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
        
        # Copy run.sh script to experiment directory for parallel execution
        run_script_src = self.project_root / 'run.sh'
        run_script_dst = experiment_dir / 'run.sh'
        if run_script_src.exists():
            shutil.copy2(run_script_src, run_script_dst)
            # Make it executable
            run_script_dst.chmod(0o755)
        
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
        
        # Call the data generation script for each enabled evaluation type
        # This ensures both results.csv and results_lib.csv are generated when both are enabled
        for evaluation_type in enabled_evaluations:
            cmd = [
                '/home/mk422/miniconda3/bin/conda', 'run', '-n', conda_env, 'python3', str(generator_script)
            ] + size_args + ['--data-type', data_type, '--output-dir', str(benchmark_dir), '--evaluation-type', evaluation_type]
            
            self.logger.info(f"Running data generation for {benchmark_name} ({evaluation_type}) in {conda_env} environment")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Data generation failed for {benchmark_name} ({evaluation_type}): {result.stderr}")
                raise RuntimeError(f"Data generation failed for {benchmark_name} ({evaluation_type})")
        else:
                self.logger.info(f"Successfully generated {benchmark_name} data for {evaluation_type}")
        
        # If no evaluation types are enabled, run without evaluation type (generates all files)
        if not enabled_evaluations:
            cmd = [
                '/home/mk422/miniconda3/bin/conda', 'run', '-n', conda_env, 'python3', str(generator_script)
            ] + size_args + ['--data-type', data_type, '--output-dir', str(benchmark_dir)]
            
            self.logger.info(f"Running data generation for {benchmark_name} (all types) in {conda_env} environment")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"Data generation failed for {benchmark_name}: {result.stderr}")
                raise RuntimeError(f"Data generation failed for {benchmark_name}")
            else:
                self.logger.info(f"Successfully generated {benchmark_name} data for all evaluation types")
    
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
        
        # Determine which evaluation types are enabled
        eval_config = self.config['evaluation']['analysis_styles']
        enabled_evaluations = []
        if eval_config['control']['enabled']:
            enabled_evaluations.append('control')
        if eval_config['library']['enabled']:
            enabled_evaluations.append('library')
        if eval_config['rounding']['enabled']:
            enabled_evaluations.append('rounding')
        
        # Call the data generation script for each enabled evaluation type
        # This ensures both results.csv and results_lib.csv are generated when both are enabled
        for evaluation_type in enabled_evaluations:
            cmd = [
                '/home/mk422/miniconda3/bin/conda', 'run', '-n', conda_env, 'python3', str(generator_script),
                '--iterations', str(iterations), '--data-type', data_type, '--output-root', str(benchmark_dir),
                '--evaluation-type', evaluation_type
            ]
            
            # Add compare flag if enabled in config
            if config.get('compare', False):
                cmd.append('--compare')
            
            # Add individual_benchmarks flag if enabled in config
            individual_benchmarks = config.get('individual_benchmarks', True)
            cmd.extend(['--individual-benchmarks', str(individual_benchmarks).lower()])
            
            # Add all_benchmarks flag if enabled in config
            all_benchmarks = config.get('all_benchmarks', True)
            cmd.extend(['--all-benchmarks', str(all_benchmarks).lower()])
            
            # Add sub-benchmarks from config
            if 'sub_benchmarks' in config:
                cmd.extend(['--sub-benchmarks'] + config['sub_benchmarks'])
            
            self.logger.info(f"Running PSA-CMA-ES data generation ({evaluation_type}) in {conda_env} environment")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"PSA-CMA-ES data generation failed ({evaluation_type}): {result.stderr}")
                raise RuntimeError(f"PSA-CMA-ES data generation failed ({evaluation_type})")
            else:
                self.logger.info(f"Successfully generated PSA-CMA-ES data for {evaluation_type}")
        
        # If no evaluation types are enabled, run without evaluation type (generates all files)
        if not enabled_evaluations:
            cmd = [
                '/home/mk422/miniconda3/bin/conda', 'run', '-n', conda_env, 'python3', str(generator_script),
                '--iterations', str(iterations), '--data-type', data_type, '--output-root', str(benchmark_dir)
            ]
            
            # Add compare flag if enabled in config
            if config.get('compare', False):
                cmd.append('--compare')
            
            # Add individual_benchmarks flag if enabled in config
            individual_benchmarks = config.get('individual_benchmarks', True)
            cmd.extend(['--individual-benchmarks', str(individual_benchmarks).lower()])
            
            # Add all_benchmarks flag if enabled in config
            all_benchmarks = config.get('all_benchmarks', True)
            cmd.extend(['--all-benchmarks', str(all_benchmarks).lower()])
            
            # Add sub-benchmarks from config
            if 'sub_benchmarks' in config:
                cmd.extend(['--sub-benchmarks'] + config['sub_benchmarks'])
            
            self.logger.info(f"Running PSA-CMA-ES data generation (all types) in {conda_env} environment")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                self.logger.error(f"PSA-CMA-ES data generation failed: {result.stderr}")
                raise RuntimeError("PSA-CMA-ES data generation failed")
            else:
                self.logger.info("Successfully generated PSA-CMA-ES data for all evaluation types")
    
    def _generate_model_data(self, config: Dict, datasets_dir: Path):
        """Generate model-specific data."""
        # This would be implemented based on specific model requirements
        self.logger.warning("Model data generation not yet implemented")
    

    
    def run_evaluations(self):
        """Generate TODO.txt with evaluation commands for parallel execution."""
        self.logger.info("Generating evaluation commands for parallel execution...")
        
        eval_config = self.config['evaluation']
        datasets_dir = self.experiment_dir / 'Datasets'
        results_dir = self.experiment_dir / 'Results'
        
        # Create TODO.txt file with evaluation commands
        todo_path = self.experiment_dir / 'TODO.txt'
        
        # Create separate TODO files for GPU and CPU algorithms
        gpu_todo_file = self.experiment_dir / "TODO_GPU.txt"
        cpu_todo_file = self.experiment_dir / "TODO_CPU.txt"
        
        # Initialize the files
        gpu_todo_file.write_text("")
        cpu_todo_file.write_text("")
        
        # Generate commands for each enabled analysis style
        for style_name, style_config in eval_config['analysis_styles'].items():
            if not style_config['enabled']:
                continue
                
            self.logger.info(f"Generating commands for {style_name} evaluation...")
            self._generate_evaluation_commands(style_name, datasets_dir, results_dir, todo_path, gpu_todo_file, cpu_todo_file)
        
        # Write the combined TODO file for backward compatibility
        with open(todo_path, 'w') as f:
            f.write("# Combined TODO List - GPU and CPU Algorithms\n")
            f.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Experiment: {self.config['experiment']['name']}\n\n")
            
            # Add GPU algorithms section
            f.write("# GPU Algorithms\n")
            f.write("# ==============\n")
            if gpu_todo_file.exists():
                f.write(gpu_todo_file.read_text())
            
            # Add CPU algorithms section
            f.write("# CPU Algorithms\n")
            f.write("# ==============\n")
            if cpu_todo_file.exists():
                f.write(cpu_todo_file.read_text())
        
        self.logger.info(f"Generated TODO files:")
        self.logger.info(f"  - GPU algorithms: {gpu_todo_file}")
        self.logger.info(f"  - CPU algorithms: {cpu_todo_file}")
        self.logger.info(f"  - Combined: {todo_path}")
        self.logger.info("Run './run.sh [NUM_CPUS]' in the experiment directory to execute evaluations in parallel")
    
    def _generate_evaluation_commands(self, style_name: str, datasets_dir: Path, results_dir: Path, todo_file_path, gpu_todo_file, cpu_todo_file):
        """Generate evaluation commands for individual algorithms based on script contents."""
        scripts_dir = self.project_root / 'Scripts'
        
        # Get noise level from config
        noise_level = self.config['data_generation']['noise']['level']
        
        # Define which algorithms are in each evaluation type, with GPU/CPU classification
        algorithm_sets = {
            'control': {
                'gpu': [
                    ('deepsr', 'unrounded/control_library/deepsr.sh'),
                    ('kan', 'unrounded/control_library/kan.sh'),
                    ('e2e_transformer', 'unrounded/control_library/e2e_transformer.sh'),
                    ('tpsr', 'unrounded/control_library/tpsr.sh')
                ],
                'cpu': [
                    ('pysr', 'unrounded/control_library/pysr.sh'),
                    ('qlattice', 'unrounded/control_library/qlattice.sh'),
                    ('linear', 'unrounded/control_library/linear.sh')
                ]
            },
            'library': {
                'gpu': [
                    ('deepsr', 'unrounded/algorithm_library/deepsr_lib.sh'),
                    ('tpsr', 'unrounded/algorithm_library/tpsr_lib.sh')
                ],
                'cpu': [
                    ('pysr', 'unrounded/algorithm_library/pysr_lib.sh'),
                    ('linear', 'unrounded/algorithm_library/linear_lib.sh')
                ]
            },
            'rounding': {
                'gpu': [
                    ('deepsr', 'rounded/deepsr.sh'),
                    ('tpsr', 'rounded/tpsr.sh')
                ],
                'cpu': [
                    ('pysr', 'rounded/pysr.sh'),
                    ('linear', 'rounded/linear.sh')
                ]
            }
        }
        
        # Get the algorithms for this evaluation type
        algorithms = algorithm_sets.get(style_name, {'gpu': [], 'cpu': []})
        
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
                    self.logger.info(f"Adding individual algorithm commands for {benchmark_name} {data_type} data: {csv_path}")
                    
                    # Generate commands for GPU algorithms
                    for alg_name, script_path in algorithms['gpu']:
                        full_script_path = scripts_dir / script_path
                        
                        # Use relative path for CSV file since run.sh will be executed from experiment directory
                        relative_csv_path = csv_path.relative_to(self.experiment_dir)
                        
                        if style_name == 'control':
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                        elif style_name == 'library':
                            problem_type = 'leading_ones' if benchmark_name == 'leadingones' else 'one_max'
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{problem_type}" "{noise_level}"'
                        elif style_name == 'rounding':
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                        else:
                            raise ValueError(f"Unknown analysis style: {style_name}")
                        
                        with open(gpu_todo_file, 'a') as f:
                            f.write(f"# {style_name} {alg_name} for {benchmark_name}\n{cmd}\n\n")
                    
                    # Generate commands for CPU algorithms
                    for alg_name, script_path in algorithms['cpu']:
                        full_script_path = scripts_dir / script_path
                        
                        # Use relative path for CSV file since run.sh will be executed from experiment directory
                        relative_csv_path = csv_path.relative_to(self.experiment_dir)
                        
                        if style_name == 'control':
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                        elif style_name == 'library':
                            problem_type = 'leading_ones' if benchmark_name == 'leadingones' else 'one_max'
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{problem_type}" "{noise_level}"'
                        elif style_name == 'rounding':
                            cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                        else:
                            raise ValueError(f"Unknown analysis style: {style_name}")
                        
                        with open(cpu_todo_file, 'a') as f:
                            f.write(f"# {style_name} {alg_name} for {benchmark_name}\n{cmd}\n\n")
                else:
                    self.logger.warning(f"CSV file not found: {csv_path}")
            
            elif benchmark_name == 'psacmaes':
                # Handle PSA-CMA-ES benchmarks
                # Only generate commands for individual benchmarks if they are enabled
                if benchmark_config.get('individual_benchmarks', True):
                    for sub_benchmark in benchmark_config['sub_benchmarks']:
                        # PSA-CMA-ES creates subdirectories with lowercase names
                        sub_benchmark_lower = sub_benchmark.lower().replace('_', '')
                        csv_filename = "psa_vars.csv"  # The actual data file
                        csv_path = datasets_dir / 'PSACMAES' / sub_benchmark_lower / csv_filename
                        
                        if csv_path.exists():
                            self.logger.info(f"Adding individual algorithm commands for PSA-CMA-ES {sub_benchmark} {data_type} data: {csv_path}")
                            
                            # Generate commands for GPU algorithms
                            for alg_name, script_path in algorithms['gpu']:
                                full_script_path = scripts_dir / script_path
                                
                                # Use relative path for CSV file since run.sh will be executed from experiment directory
                                relative_csv_path = csv_path.relative_to(self.experiment_dir)
                                
                                if style_name == 'control':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                elif style_name == 'library':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                                elif style_name == 'rounding':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                else:
                                    raise ValueError(f"Unknown analysis style: {style_name}")
                                
                                with open(gpu_todo_file, 'a') as f:
                                    f.write(f"# {style_name} {alg_name} for PSA-CMA-ES {sub_benchmark}\n{cmd}\n\n")
                            
                            # Generate commands for CPU algorithms
                            for alg_name, script_path in algorithms['cpu']:
                                full_script_path = scripts_dir / script_path
                                
                                # Use relative path for CSV file since run.sh will be executed from experiment directory
                                relative_csv_path = csv_path.relative_to(self.experiment_dir)
                                
                                if style_name == 'control':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                elif style_name == 'library':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                                elif style_name == 'rounding':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                else:
                                    raise ValueError(f"Unknown analysis style: {style_name}")
                                
                                with open(cpu_todo_file, 'a') as f:
                                    f.write(f"# {style_name} {alg_name} for PSA-CMA-ES {sub_benchmark}\n{cmd}\n\n")
                        else:
                            self.logger.warning(f"CSV file not found: {csv_path}")
                
                # Handle aggregated PSA-CMA-ES data if enabled
                if benchmark_config.get('all_benchmarks', False):
                    csv_path = datasets_dir / 'PSACMAES' / 'all_benchmarks.csv'
                    
                    if csv_path.exists():
                        self.logger.info(f"Adding individual algorithm commands for PSA-CMA-ES aggregated data: {csv_path}")
                        
                        # Generate commands for GPU algorithms
                        for alg_name, script_path in algorithms['gpu']:
                            full_script_path = scripts_dir / script_path
                            relative_csv_path = csv_path.relative_to(self.experiment_dir)
                            
                            if style_name == 'control':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            elif style_name == 'library':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                            elif style_name == 'rounding':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            else:
                                raise ValueError(f"Unknown analysis style: {style_name}")
                            
                            with open(gpu_todo_file, 'a') as f:
                                f.write(f"# {style_name} {alg_name} for PSA-CMA-ES all_benchmarks\n{cmd}\n\n")
                        
                        # Generate commands for CPU algorithms
                        for alg_name, script_path in algorithms['cpu']:
                            full_script_path = scripts_dir / script_path
                            relative_csv_path = csv_path.relative_to(self.experiment_dir)
                            
                            if style_name == 'control':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            elif style_name == 'library':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                            elif style_name == 'rounding':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            else:
                                raise ValueError(f"Unknown analysis style: {style_name}")
                            
                            with open(cpu_todo_file, 'a') as f:
                                f.write(f"# {style_name} {alg_name} for PSA-CMA-ES all_benchmarks\n{cmd}\n\n")
                    else:
                        self.logger.warning(f"CSV file not found: {csv_path}")
                
                # Handle leave-one-out analysis if enabled
                if benchmark_config.get('compare', False):
                    for sub_benchmark in benchmark_config['sub_benchmarks']:
                        sub_benchmark_lower = sub_benchmark.lower().replace('_', '')
                        csv_path = datasets_dir / 'PSACMAES' / 'absent' / sub_benchmark_lower / 'psa_vars.csv'
                        
                        if csv_path.exists():
                            self.logger.info(f"Adding individual algorithm commands for PSA-CMA-ES leave-one-out (excluding {sub_benchmark}): {csv_path}")
                            
                            # Generate commands for GPU algorithms
                            for alg_name, script_path in algorithms['gpu']:
                                full_script_path = scripts_dir / script_path
                                relative_csv_path = csv_path.relative_to(self.experiment_dir)
                                
                                if style_name == 'control':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                elif style_name == 'library':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                                elif style_name == 'rounding':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                else:
                                    raise ValueError(f"Unknown analysis style: {style_name}")
                                
                                with open(gpu_todo_file, 'a') as f:
                                    f.write(f"# {style_name} {alg_name} for PSA-CMA-ES leave-one-out (excluding {sub_benchmark})\n{cmd}\n\n")
                            
                            # Generate commands for CPU algorithms
                            for alg_name, script_path in algorithms['cpu']:
                                full_script_path = scripts_dir / script_path
                                relative_csv_path = csv_path.relative_to(self.experiment_dir)
                                
                                if style_name == 'control':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                elif style_name == 'library':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "psa" "{noise_level}"'
                                elif style_name == 'rounding':
                                    cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                                else:
                                    raise ValueError(f"Unknown analysis style: {style_name}")
                                
                                with open(cpu_todo_file, 'a') as f:
                                    f.write(f"# {style_name} {alg_name} for PSA-CMA-ES leave-one-out (excluding {sub_benchmark})\n{cmd}\n\n")
                        else:
                            self.logger.warning(f"CSV file not found: {csv_path}")
            
            elif benchmark_name == 'model':
                # Handle model-based benchmarks
                for instance_size in benchmark_config['instance_sizes']:
                    csv_filename = f"model_data_{instance_size}.csv"
                    csv_path = datasets_dir / 'Model' / csv_filename
                    
                    if csv_path.exists():
                        self.logger.info(f"Adding individual algorithm commands for model data {instance_size}: {csv_path}")
                        
                        # Generate commands for GPU algorithms
                        for alg_name, script_path in algorithms['gpu']:
                            full_script_path = scripts_dir / script_path
                            relative_csv_path = csv_path.relative_to(self.experiment_dir)
                            
                            if style_name == 'control':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            elif style_name == 'library':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "model" "{noise_level}"'
                            elif style_name == 'rounding':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            else:
                                raise ValueError(f"Unknown analysis style: {style_name}")
                            
                            with open(gpu_todo_file, 'a') as f:
                                f.write(f"# {style_name} {alg_name} for model data {instance_size}\n{cmd}\n\n")
                        
                        # Generate commands for CPU algorithms
                        for alg_name, script_path in algorithms['cpu']:
                            full_script_path = scripts_dir / script_path
                            relative_csv_path = csv_path.relative_to(self.experiment_dir)
                            
                            if style_name == 'control':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            elif style_name == 'library':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "model" "{noise_level}"'
                            elif style_name == 'rounding':
                                cmd = f'bash "{full_script_path}" "{relative_csv_path}" "{noise_level}"'
                            else:
                                raise ValueError(f"Unknown analysis style: {style_name}")
                            
                            with open(cpu_todo_file, 'a') as f:
                                f.write(f"# {style_name} {alg_name} for model data {instance_size}\n{cmd}\n\n")
                    else:
                        self.logger.warning(f"CSV file not found: {csv_path}")
        
        self.logger.info(f"Added {style_name} evaluation commands to TODO files")
    
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