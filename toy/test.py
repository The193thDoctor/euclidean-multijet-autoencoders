import torch
import torch.nn.functional as F
import time

# Example tensor and convolution parameters
tensor = torch.randn(1, 1, 512, 10)  # A large 4D tensor
kernel = torch.randn(1, 1, 3, 3)  # Convolution kernel



# Method 1: Split, Convolve, Combine
half_size = tensor.size(-1) // 2
half1 = tensor
half2 = torch.flip(half1, dims=[3])
start_time = time.time()
conv1, conv2 = None, None
for _ in range(100):
    conv1 = F.conv2d(half1, kernel)
    conv2 = F.conv2d(half2, kernel)
result1 = torch.cat((conv1, conv2), dim=-1)
method1_time = time.time() - start_time

print(result1.shape)

# Method 2: Convolve Entire Tensor
start_time = time.time()

combined = torch.cat((half1, half2), dim=-1)
result2 = None
for _ in range(100):
    result2 = F.conv2d(combined, kernel)
method2_time = time.time() - start_time
print(result2.shape)

print(f"Method 1 Time: {method1_time:.6f} seconds")
print(f"Method 2 Time: {method2_time:.6f} seconds")