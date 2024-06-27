import torch
from PIL import Image, ImageDraw
import cupy as np
from AI_model import WaldoRecognizer
from torchvision import transforms

def divide_image_into_patches(image_path, patch_size=64):
    # Open the image
    image = Image.open(image_path)
    image = image.convert('RGB')  # Ensure image is in RGB format
    
    # Get image dimensions
    width, height = image.size
    
    # Calculate the number of patches in each dimension
    num_patches_x = width // patch_size
    num_patches_y = height // patch_size
    
    # Initialize a list to store patches
    patches = []
    
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            # Define the box to crop the image
            box = (j * patch_size, i * patch_size, (j + 1) * patch_size, (i + 1) * patch_size)
            patch = image.crop(box)
            patches.append(np.array(patch))
    
    # Convert list of patches to a NumPy array and then to a PyTorch tensor
    patches_array = np.array(patches)
    patches_tensor = torch.tensor(patches_array, dtype=torch.float32)
    
    # Rearrange dimensions to match the format (num_patches, channels, height, width)
    patches_tensor = patches_tensor.permute(0, 3, 1, 2)
    
    return patches_tensor, num_patches_x, num_patches_y

def load_model(model_path):
    # Load the model
    network = WaldoRecognizer()
    network.load_state_dict(torch.load(model_path))
    network.eval()  # Set the model to evaluation mode
    return network

def feed_patches_to_model(model, patches_tensor):
    # Ensure patches_tensor is on the same device as the model (CPU or GPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    patches_tensor = patches_tensor.to(device)


    to_tensor = transforms.Compose([
        transforms.Normalize((0, 0, 0), (1, 1, 1))  # Normalize image pixels
    ])

    transformed_patches_tensor = to_tensor(patches_tensor)
    
    # Feed the patches to the model
    with torch.no_grad():
        outputs = model(transformed_patches_tensor)
    
    return outputs

def find_highest_output_patch(outputs):
    # Find the index of the patch with the highest value at index 1
    highest_index = torch.argmax(outputs[:, 1])
    return highest_index.item()

def calculate_patch_coordinates(patch_index, num_patches_x, patch_size=64):
    # Calculate the coordinates of the patch in the original image
    row = patch_index // num_patches_x
    col = patch_index % num_patches_x
    x_min = col * patch_size
    y_min = row * patch_size
    x_max = x_min + patch_size
    y_max = y_min + patch_size
    return (x_min, y_min, x_max, y_max)

def draw_rectangles(image_path, coordinates):
    # Open the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)
    
    # Draw a green rectangle around the patch
    for coordinate in coordinates:
        draw.rectangle(coordinate, outline="red", width=10)
    
    # Show the image
    image.show()

# Example usage
image_path = 'Data/original-images/1.jpg'
model_path = 'models/9997-v2_model-xxl.pth' 

# Step 1: Divide the image into patches
patches_tensor, num_patches_x, num_patches_y = divide_image_into_patches(image_path)

# Step 2: Load the model
model = load_model(model_path)

# Step 3: Feed the patches to the model
outputs = feed_patches_to_model(model, patches_tensor)

# Step 4: Find the patch with the highest output at index 0
total_found = 0
highest_patch_index = find_highest_output_patch(outputs)
all_squares = []
for i, output in enumerate(outputs):
    if output[0].item() < output[1].item():
        total_found += 1
        coordinates = calculate_patch_coordinates(i, num_patches_x)
        # print(output)
        all_squares.append(coordinates)

draw_rectangles(image_path, all_squares)   
print("Total Waldos found:", total_found)