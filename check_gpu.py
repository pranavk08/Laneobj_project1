import torch

print('='*50)
print('GPU Detection Status')
print('='*50)
print(f'CUDA Available: {torch.cuda.is_available()}')
print(f'CUDA Version: {torch.version.cuda}')
print(f'PyTorch Version: {torch.__version__}')
print(f'Number of GPUs: {torch.cuda.device_count()}')

if torch.cuda.is_available():
    print(f'Current GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB')
    print(f'GPU Compute Capability: {torch.cuda.get_device_capability(0)}')
    print('='*50)
    
    # Test computation
    print('Testing GPU computation...')
    x = torch.randn(1000, 1000).cuda()
    y = x @ x
    print(f'Test Computation Device: {y.device}')
    print('GPU is WORKING ✓')
else:
    print('='*50)
    print('GPU is NOT detected ✗')
    print('Running on CPU only')
