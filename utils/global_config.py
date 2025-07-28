"""
Global configuration utilities for symbolic regression algorithms.
This module provides GPU detection and TensorFlow device configuration.
"""

import os

# Try to import TensorFlow, but handle the case where it's not available
try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("Warning: TensorFlow not available, using CPU fallback")

def is_gpu_enabled():
    """
    Check if GPU is enabled for TensorFlow.
    
    Returns:
        bool: True if GPU is available and enabled, False otherwise
    """
    if not TENSORFLOW_AVAILABLE:
        return False
    
    try:
        # Check if TensorFlow can see any GPUs
        gpus = tf.config.list_physical_devices('GPU')
        return len(gpus) > 0
    except Exception:
        # If TensorFlow is not available or there's an error, assume no GPU
        return False

def get_tensorflow_device():
    """
    Get the TensorFlow device configuration.
    
    Returns:
        str: Device string ('GPU:0' if GPU available, '/CPU:0' otherwise)
    """
    try:
        if is_gpu_enabled():
            return 'GPU:0'
        else:
            return '/CPU:0'
    except Exception:
        return '/CPU:0'

def get_device_preference():
    """
    Get device preference for algorithms that support both CPU and GPU.
    
    Returns:
        str: 'gpu' if GPU is available, 'cpu' otherwise
    """
    try:
        if is_gpu_enabled():
            return 'gpu'
        else:
            return 'cpu'
    except Exception:
        return 'cpu'

def set_memory_growth():
    """
    Set TensorFlow GPU memory growth to prevent memory allocation issues.
    This is useful for preventing GPU memory conflicts in parallel execution.
    """
    if not TENSORFLOW_AVAILABLE:
        return
    
    try:
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth enabled")
    except Exception as e:
        print(f"Warning: Could not set GPU memory growth: {e}")

# Initialize GPU memory growth on import (only if TensorFlow is available)
if TENSORFLOW_AVAILABLE:
    set_memory_growth() 