import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import torchvision.transforms as transforms
import random
import os

def verify_class_balance(x_train, y_train, samples_per_class=400, augment_if_needed=True):
    balanced_x = []
    balanced_y = []
    
    augment = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.ToTensor(), 
    ])
    
    for digit in range(10):
        indices = np.where(y_train == digit)[0]
        available_samples = len(indices)
        
        if available_samples >= samples_per_class:

            selected_indices = np.random.choice(indices, samples_per_class, replace=False)
            balanced_x.extend([x_train[i] for i in selected_indices])
            balanced_y.extend([y_train[i] for i in selected_indices])
        else:

            balanced_x.extend([x_train[i] for i in indices])
            balanced_y.extend([y_train[i] for i in indices])
            
            if augment_if_needed:

                needed = samples_per_class - available_samples
                for _ in range(needed):
                    idx = random.choice(indices)
                    img = x_train[idx]
 
                    img_pil = Image.fromarray((img * 255).astype(np.uint8))
                    augmented = augment(img_pil).numpy()[0]  # Shape: [1, 28, 28] -> [28, 28]
                    balanced_x.append(augmented)
                    balanced_y.append(y_train[idx])
    
    return np.array(balanced_x), np.array(balanced_y)

def add_noise(image, noise_factor=0.1):
    noise = np.random.normal(loc=0.0, scale=noise_factor, size=image.shape)
    noisy_image = image + noise
    return np.clip(noisy_image, 0.0, 1.0)

def flatten_mnist_for_roblox(
    output_file="MNISTData.lua",
    max_train_samples=8000,
    max_test_samples=4000,
    samples_per_class=400,
    noise_factor=0.1,
    normalize_mean=0.1307,
    normalize_std=0.3081
):

    if max_train_samples <= 0 or max_test_samples <= 0:
        raise ValueError("max_train_samples and max_test_samples must be positive")
    if samples_per_class <= 0:
        raise ValueError("samples_per_class must be positive")
    
    try:
        (x_train, y_train), (x_test, y_test) = mnist.load_data()
    except Exception as e:
        raise RuntimeError(f"Failed to load MNIST dataset: {e}")
    
    # Normalize to [0,1]
    x_train = x_train.astype(np.float32) / 255.0
    x_test = x_test.astype(np.float32) / 255.0
    
    print("Balancing training classes...")
    x_train, y_train = verify_class_balance(x_train, y_train, samples_per_class=samples_per_class)
    
    print("Balancing test classes...")
    x_test, y_test = verify_class_balance(x_test, y_test, samples_per_class=max_test_samples // 10)
    
    x_train = (x_train - normalize_mean) / normalize_std
    x_test = (x_test - normalize_mean) / normalize_std
    
    max_train_samples = min(max_train_samples, len(x_train))
    max_test_samples = min(max_test_samples, len(x_test))
    
    print(f"Writing {max_train_samples} training samples and {max_test_samples} testing samples to {output_file}...")
    
    try:
        with open(output_file, "w") as f:
            f.write("local MNISTData = {}\n\n")
            
            f.write("MNISTData.Training = {\n")
            for i in range(max_train_samples):
                image = x_train[i]
                image = add_noise(image, noise_factor=noise_factor)
                label = int(y_train[i])
                
                f.write("\t{\n")
                f.write(f"\t\tlabel = {label},\n")
                f.write("\t\timage = {")
                pixel_vals = [f"{val:.3f}" for val in image.flatten()]
                f.write(",".join(pixel_vals))
                f.write("},\n")
                f.write("\t},\n")
            f.write("}\n\n")
            
            f.write("MNISTData.Testing = {\n")
            for i in range(max_test_samples):
                image = x_test[i]
                image = add_noise(image, noise_factor=noise_factor * 0.5)
                label = int(y_test[i])
                
                f.write("\t{\n")
                f.write(f"\t\tlabel = {label},\n")
                f.write("\t\timage = {")
                pixel_vals = [f"{val:.3f}" for val in image.flatten()]
                f.write(",".join(pixel_vals))
                f.write("},\n")
                f.write("\t},\n")
            f.write("}\n\n")
            
            f.write("""
local function downsample_28x28_to_7x7(image)
\tlocal result = {}
\tfor i = 1, 7 do
\t\tfor j = 1, 7 do
\t\t\tlocal sum = 0
\t\t\tlocal count = 0
\t\t\tfor di = 0, 3 do
\t\t\t\tfor dj = 0, 3 do
\t\t\t\t\tlocal row = (i-1)*4 + di + 1
\t\t\t\t\tlocal col = (j-1)*4 + dj + 1
\t\t\t\t\tlocal index = (row-1)*28 + col
\t\t\t\t\tif image[index] then
\t\t\t\t\t\tsum = sum + image[index]
\t\t\t\t\t\tcount = count + 1
\t\t\t\t\tend
\t\t\t\tend
\t\t\tend
\t\t\tresult[(i-1)*7 + j] = count > 0 and sum / count or 0
\t\tend
\tend
\treturn result
end

for _, sample in ipairs(MNISTData.Training) do
\tsample.image_7x7 = downsample_28x28_to_7x7(sample.image)
end

for _, sample in ipairs(MNISTData.Testing) do
\tsample.image_7x7 = downsample_28x28_to_7x7(sample.image)
end

function MNISTData.getRandomTrainingSample()
\tlocal index = math.random(1, #MNISTData.Training)
\treturn MNISTData.Training[index]
end

function MNISTData.getRandomTestSample()
\tlocal index = math.random(1, #MNISTData.Testing)
\treturn MNISTData.Testing[index]
end

function MNISTData.getAllTrainingSamples()
\treturn MNISTData.Training
end

function MNISTData.getAllTestSamples()
\treturn MNISTData.Testing
end

return MNISTData
""")
        
        print(f"Successfully saved MNIST dataset to {output_file}")
  
    except IOError as e:
        raise RuntimeError(f"Failed to write to {output_file}: {e}")

if __name__ == "__main__":
    flatten_mnist_for_roblox(
        output_file="MNISTData.lua",
        max_train_samples=8000,
        max_test_samples=4000,
        samples_per_class=400,
        noise_factor=0.1,
        normalize_mean=0.1307,
        normalize_std=0.3081
    )
