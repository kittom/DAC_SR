#!/usr/bin/env python3
"""
Debug script for TPSR to identify the exact error
"""

import sys
import os
import pandas as pd
import numpy as np
import time
import signal

# Change to TPSR directory first
os.chdir("SR_algorithms/TPSR")
print(f"Changed to directory: {os.getcwd()}")

try:
    from parsers import get_parser
    from symbolicregression.envs import build_env
    from symbolicregression.model import build_modules
    from symbolicregression.trainer import Trainer
    from symbolicregression.e2e_model import Transformer, pred_for_sample_no_refine
    from tpsr import tpsr_fit
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def test_tpsr():
    print("=== TPSR Debug Test ===")
    
    # Test data
    X = np.array([[1, 2], [3, 4], [5, 6]])
    y = np.array([[3], [7], [11]])
    
    print(f"Test data shape: X={X.shape}, y={y.shape}")
    
    try:
        # Step 1: Parse parameters
        print("\n1. Parsing parameters...")
        parser = get_parser()
        tpsr_params = parser.parse_args([])
        tpsr_params.cpu = True
        tpsr_params.device = 'cpu'
        tpsr_params.debug = False
        tpsr_params.backbone_model = 'e2e'
        tpsr_params.horizon = 10
        tpsr_params.width = 2
        tpsr_params.rollout = 2
        tpsr_params.ucb_constant = 0.5
        tpsr_params.ucb_base = 5.0
        tpsr_params.print_freq = 10
        print("✓ Parameters parsed")
        
        # Step 2: Check model file
        print("\n2. Checking model file...")
        model_path = "./symbolicregression/weights/model.pt"
        if not os.path.exists(model_path):
            print(f"✗ Model file not found: {model_path}")
            print(f"Current directory: {os.getcwd()}")
            return False
        print(f"✓ Model file found: {model_path}")
        print(f"  Size: {os.path.getsize(model_path) / (1024*1024):.1f} MB")
        
        # Step 3: Build environment
        print("\n3. Building equation environment...")
        equation_env = build_env(tpsr_params)
        print("✓ Environment built")
        
        # Step 4: Build modules
        print("\n4. Building modules...")
        modules = build_modules(equation_env, tpsr_params)
        print("✓ Modules built")
        
        # Step 5: Create trainer
        print("\n5. Creating trainer...")
        trainer = Trainer(modules, equation_env, tpsr_params)
        print("✓ Trainer created")
        
        # Step 6: Test Transformer creation
        print("\n6. Testing Transformer creation...")
        samples = {'x_to_fit': [X], 'y_to_fit': [y]}
        transformer = Transformer(params=tpsr_params, env=equation_env, samples=samples)
        print("✓ Transformer created")
        
        # Step 7: Test tpsr_fit
        print("\n7. Testing tpsr_fit...")
        start_time = time.time()
        final_seq, time_elapsed, best_reward = tpsr_fit([X], [y], tpsr_params, equation_env)
        total_time = time.time() - start_time
        print(f"✓ tpsr_fit completed in {total_time:.2f} seconds")
        print(f"  Final sequence length: {len(final_seq) if final_seq else 0}")
        print(f"  Best reward: {best_reward:.6f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_tpsr()
    if success:
        print("\n✅ TPSR test completed successfully!")
    else:
        print("\n❌ TPSR test failed!")
        sys.exit(1) 