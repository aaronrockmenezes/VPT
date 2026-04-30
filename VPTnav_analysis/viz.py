import torch
import matplotlib.pyplot as plt
import numpy as np

data_path = "/users/arock3/scratch/VPT1_DATA/v18_rgb_cam/data/data_node0_gpu0/cam/Yes/env_21/cam_rgb_save.pt"

# Load the PyTorch tensor
image_tensor = torch.load(data_path, map_location=torch.device('cpu'))

# Ensure the tensor is on the CPU and detach it from the graph if necessary
if image_tensor.is_cuda:
    image_tensor = image_tensor.cpu()

image_array = image_tensor.detach().numpy()

print(np.max(image_array))
print(np.min(image_array))

# Ensure the shape is exactly what matplotlib expects (H, W, C)
# The prompt mentions it is already 256x256x3, so no transposition is needed.

# Save the image using plt
output_filename = "/users/arock3/data/arock3/VPT/VPTnav_analysis/saved_cam_rgb.png"
plt.imsave(output_filename, image_array)

print(f"Image successfully saved to {output_filename}")