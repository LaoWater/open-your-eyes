import torch
import torch.nn as nn
import torchvision
import cv2
import numpy as np
import time
import threading
from queue import Queue, Empty
from collections import deque
import psutil
import GPUtil
from typing import List, Dict, Tuple, Optional, Any
import onnxruntime as ort
from concurrent.futures import ThreadPoolExecutor
import multiprocessing as mp

class ModelOptimizer:
    """
    Comprehensive model optimization for production deployment
    
    Real-world goal: Reduce inference time from 200ms to <50ms
    Critical for real-time restaurant safety monitoring
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        self.model_path = model_path
        self.device = device
        self.optimized_models = {}
        
    def optimize_pytorch_model(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 416, 416)):
        """
        Optimize PyTorch model for production inference
        
        Techniques applied:
        1. TorchScript compilation (JIT)
        2. Model quantization
        3. Operator fusion
        4. Memory optimization
        """
        print("🔧 Optimizing PyTorch Model...")
        
        # 1. TorchScript Optimization (15-30% speedup)
        model.eval()
        
        # Create sample input for tracing
        sample_input = torch.randn(*input_shape).to(self.device)
        
        # Warm up the model
        with torch.no_grad():
            for _ in range(10):
                _ = model(sample_input)
        
        # Trace the model
        traced_model = torch.jit.trace(model, sample_input)
        
        # Optimize the traced model
        traced_model = torch.jit.optimize_for_inference(traced_model)
        
        # 2. Quantization (2-4x speedup, but slightly less accurate)
        quantized_model = None
        try:
            # Dynamic quantization (easiest to implement)
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},  # Quantize these layer types
                dtype=torch.qint8
            )
            print("✅ Dynamic quantization successful")
        except Exception as e:
            print(f"⚠️  Quantization failed: {e}")
        
        # 3. Memory Optimization
        traced_model.to(memory_format=torch.channels_last)
        
        # Store optimized models
        self.optimized_models['traced'] = traced_model
        if quantized_model:
            self.optimized_models['quantized'] = quantized_model
        
        print(f"✅ PyTorch optimization complete")
        return traced_model, quantized_model
    
    def convert_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 416, 416)):
        """
        Convert PyTorch model to ONNX for cross-platform deployment
        
        Benefits:
        - 10-50% faster inference
        - Smaller model size
        - Cross-platform compatibility
        - Better optimization opportunities
        """
        print("🔄 Converting to ONNX...")
        
        model.eval()
        sample_input = torch.randn(*input_shape).to(self.device)
        
        onnx_path = self.model_path.replace('.pth', '.onnx')
        
        torch.onnx.export(
            model,
            sample_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,  # Optimize constants
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},   # Variable batch size
                'output': {0: 'batch_size'}
            },
            verbose=False
        )
        
        print(f"✅ ONNX model saved to {onnx_path}")
        return onnx_path
    
    def create_tensorrt_engine(self, onnx_path: str):
        """
        Create TensorRT engine for NVIDIA GPU optimization
        
        Benefits:
        - 2-5x faster inference on NVIDIA GPUs
        - Aggressive optimization
        - Mixed precision support
        
        Note: Requires TensorRT installation
        """
        try:
            import tensorrt as trt
            import pycuda.driver as cuda
            import pycuda.autoinit
            
            print("🚀 Creating TensorRT engine...")
            
            # TensorRT logger
            logger = trt.Logger(trt.Logger.WARNING)
            
            # Create builder
            builder = trt.Builder(logger)
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, logger)
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model_file:
                if not parser.parse(model_file.read()):
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    return None
            
            # Build engine
            config = builder.create_builder_config()
            config.max_workspace_size = 1 << 30  # 1GB
            
            # Enable optimizations
            config.set_flag(trt.BuilderFlag.FP16)  # Half precision
            config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
            
            engine = builder.build_engine(network, config)
            
            # Save engine
            engine_path = onnx_path.replace('.onnx', '.trt')