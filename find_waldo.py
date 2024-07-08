from AI_model import WaldoRecognizer
from torchvision import transforms
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import torch
import os

def divide_image_into_patches(image_path, patch_size=64):
    image = Image.open(image_path)
    image = image.convert('RGB')
    
    width, height = image.size
    
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size
    
    patches = []
    
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patch = image.crop(box)
            patches.append(patch)
           
    to_tensor = transforms.Compose([
        # transforms.RandomHorizontalFlip(),          # Random horizontal flip
        # transforms.RandomVerticalFlip(),            # Random vertical flip
        # transforms.RandomRotation(30),              # Random rotation within 30 degrees
        # transforms.RandomGrayscale(p=0.2),          # Convert image to grayscale with a probability of 20%
        # transforms.RandomPerspective(distortion_scale=0.5, p=0.5),  # Random perspective transformation
        # transforms.RandomResizedCrop(size=64, scale=(0.95, 1.0)),  # Random resized crop
        # transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.5),  # Random Gaussian blur
        transforms.ToTensor(),                      # Convert to tensor
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize with ImageNet means and stds
        transforms.Normalize(mean=[0] * 3, std=[1] * 3)  # Normalize with Normal distribution means and stds
    ])

    patches_tensor = []
    for patch in patches:
        patches_tensor.append(to_tensor(patch))

    patches_tensor = torch.stack(patches_tensor)

    return patches_tensor, num_patches_x, num_patches_y

def load_model(model_path):
    network = WaldoRecognizer()
    network.load_state_dict(torch.load(model_path))
    network.eval()  # Set the model to evaluation mode
    return network

def feed_patches_to_model(model, patches_tensor):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    patches_tensor = patches_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(patches_tensor)
    
    return outputs

def find_highest_output_patch(outputs):
    highest_index = torch.argmax(outputs[:, 1])
    return highest_index.item()

def calculate_patch_coordinates(patch_index, num_patches_x, patch_size=64):
    row = patch_index // num_patches_x
    col = patch_index % num_patches_x
    x_min = col * patch_size
    y_min = row * patch_size
    x_max = x_min + patch_size
    y_max = y_min + patch_size
    return (x_min, y_min, x_max, y_max)

def draw_rectangles(image_path, coordinates):
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    for coordinate in coordinates:
        draw.rectangle(coordinate, outline="red", width=10)
    
    image.show()

def show_images(images):
    fig, axes = plt.subplots(1, 5, figsize=(15,6))
    to_image = transforms.ToPILImage()
    
    for i in range(len(images)):
        axes[i].imshow(to_image(images[i][0]))
        axes[i].set_title(f"Waldo : {images[i][1]}")
        axes[i].axis("off")

    plt.tight_layout()
    plt.show()

images_path = 'Data/original-images/'
model_path = 'models/best_model.pth' 
model = load_model(model_path)

for image in os.listdir(images_path):
    image_path = os.path.join(images_path, image)
    patches_tensor, num_patches_x, num_patches_y = divide_image_into_patches(image_path)

    outputs = feed_patches_to_model(model, patches_tensor)

    total_found = 0
    highest_patch_index = find_highest_output_patch(outputs)
    all_squares = []
    for i, output in enumerate(outputs):
        if output[0].item() < output[1].item():
            total_found += 1
            coordinates = calculate_patch_coordinates(i, num_patches_x)
            all_squares.append(coordinates)

    draw_rectangles(image_path, all_squares)   
    print("Total Waldos found:", total_found)