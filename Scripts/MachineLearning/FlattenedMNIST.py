import numpy as np
from tensorflow.keras.datasets import mnist
from PIL import Image
import torchvision.transforms as transforms
import random

augment = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(0, translate=(0.05, 0.05), scale=(0.95, 1.05)),
    transforms.ToTensor()
])

def add_noise(img, sigma=0.1):
    return np.clip(img + np.random.normal(0, sigma, img.shape), 0, 1)

def balance_classes(x, y, per_class, augment_missing=True):
    bx, by = [], []
    for d in range(10):
        idx = np.where(y == d)[0]
        if len(idx) >= per_class:
            chosen = np.random.choice(idx, per_class, replace=False)
            bx.extend(x[i] for i in chosen)
            by.extend(y[i] for i in chosen)
        else:
            bx.extend(x[i] for i in idx)
            by.extend(y[i] for i in idx)
            if augment_missing:
                for _ in range(per_class - len(idx)):
                    img = Image.fromarray((x[random.choice(idx)] * 255).astype(np.uint8))
                    bx.append(augment(img).numpy()[0])
                    by.append(d)

    return np.array(bx), np.array(by)

def normalize(x, mean=0.1307, std=0.3081):
    return (x - mean) / std

def write_samples(f, name, x, y, n, noise):
    f.write(f"MNISTData.{name} = {{\n")
    for i in range(n):
        img = add_noise(x[i], noise)
        lbl = int(y[i])
        vals = ",".join(f"{v:.3f}" for v in img.flatten())
        f.write(f"\t{{ label = {lbl}, image = {{{vals}}} }},\n")
    f.write("}\n\n")

def flatten_mnist(
    output_file="MNISTData.lua",
    max_train=8000,
    max_test=4000,
    per_class=400,
    noise=0.1
):
    (xt, yt), (xv, yv) = mnist.load_data()
    xt, xv = xt.astype(np.float32)/255, xv.astype(np.float32)/255

    xt, yt = balance_classes(xt, yt, per_class)
    xv, yv = balance_classes(xv, yv, max_test//10)

    xt, xv = normalize(xt), normalize(xv)
    mt, mv = min(max_train, len(xt)), min(max_test, len(xv))

    with open(output_file, "w") as f:
        f.write("local MNISTData = {}\n\n")
        write_samples(f, "Training", xt, yt, mt, noise)
        write_samples(f, "Testing", xv, yv, mv, noise * 0.5)

if __name__ == "__main__":
    flatten_mnist()
