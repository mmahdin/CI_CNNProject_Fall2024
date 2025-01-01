from google.colab import drive
import os
import zipfile
import random
import shutil

# Define paths
image_path = "/root/.cache/kagglehub/datasets/hsankesara/flickr-image-dataset/versions/1/flickr30k_images/flickr30k_images/"
base_dir = '/content/drive/MyDrive/phase3'
zip_file_name = "compressed_images.zip"

# Mount Google Drive
drive.mount('/content/drive')

# Ensure base directory exists

# Get a list of all images in the directory
all_images = [f for f in os.listdir(
    image_path) if os.path.isfile(os.path.join(image_path, f))]

# Randomly select 5000 images
selected_images = random.sample(all_images, 5000)

# Create a ZIP file
zip_file_path = os.path.join(base_dir, zip_file_name)
with zipfile.ZipFile(zip_file_path, 'w') as zipf:
    for img_name in selected_images:
        img_path = os.path.join(image_path, img_name)
        zipf.write(img_path, arcname=img_name)

# Verify file creation
if os.path.exists(zip_file_path):
    print(f"Compressed file created at: {zip_file_path}")
else:
    print("Error: Failed to create compressed file.")


# Define paths
zip_file_path = './dataset/compressed_images.zip'  # Path to the ZIP file
unzip_dir = './dataset/flicker30k'  # Directory to extract the files

# Ensure the output directory exists
os.makedirs(unzip_dir, exist_ok=True)

# Unzip the file
with zipfile.ZipFile(zip_file_path, 'r') as zipf:
    zipf.extractall(unzip_dir)


# Define the folder and caption file paths
image_folder = "./dataset/flicker30k/"
caption_file_path = "./dataset/results.csv"
output_file_path = "./dataset/captions.csv"

# Get a set of image names from the folder
image_names = set(os.listdir(image_folder))

# Filter the caption file
with open(caption_file_path, "r") as caption_file:
    lines = caption_file.readlines()

filtered_lines = []

for line in lines:
    image_name = line.split("|")[0].strip()
    if image_name in image_names:
        filtered_lines.append(line)

# Write the filtered lines to a new file
with open(output_file_path, "w") as output_file:
    output_file.writelines(filtered_lines)

print(f"Filtered captions saved to {output_file_path}")
