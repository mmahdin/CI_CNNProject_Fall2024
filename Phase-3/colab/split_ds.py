import json
import kagglehub
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


###################################

# Download latest version
path_30k = kagglehub.dataset_download("hsankesara/flickr-image-dataset")
path_30k += '/flickr30k_images'
print("Path to dataset files:", path_30k)

# Define the source directory containing images
# Replace with your actual source directory path
image_path = f'{path_30k}/flickr30k_images/'

# Define the destination directory
destination_path = './images'

# Ensure the destination directory exists
os.makedirs(destination_path, exist_ok=True)

# Get a list of all images in the directory
all_images = [f for f in os.listdir(
    image_path) if os.path.isfile(os.path.join(image_path, f))]

# Randomly select 5000 images
selected_images = all_images[:5000]

# Copy the selected images to the destination directory
for image in selected_images:
    shutil.copy(os.path.join(image_path, image),
                os.path.join(destination_path, image))

print(
    f"Successfully copied {len(selected_images)} images to '{destination_path}'.")


# Define the folder and caption file paths
image_folder = destination_path
caption_file_path = f"{path_30k}/results.csv"
output_file_path = './captions.csv'

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


###########################################################################
#                                  MSCOCO                                 #
###########################################################################


# Download latest version
path = kagglehub.dataset_download("nikhil7280/coco-image-caption")

print("Path to dataset files:", path)

caption_path = '/root/.cache/kagglehub/datasets/nikhil7280/coco-image-caption/versions/1/annotations_trainval2014/annotations/captions_train2014.json'


drive.mount('/content/drive')

base_dir = '/content/drive/MyDrive/phase3'


# Path to the captions_train2014.json file
input_file_path = caption_path
output_file_path = f'{base_dir}/coco.txt'

# Variable for the word count threshold
n = 9  # Adjust this value as needed

# Load the JSON file
with open(input_file_path, 'r') as file:
    data = json.load(file)

# Prepare the result
results = []

cnt = 0
# Process the annotations
for annotation in data['annotations']:
    caption = annotation['caption']
    word_count = len(caption.split())
    if word_count < n:
        if cnt == 10000:
            break
        cnt += 1
        if cnt % 1000 == 0:
            print(cnt)
        # Find the corresponding image
        image_id = annotation['image_id']
        image_info = next(
            img for img in data['images'] if img['id'] == image_id)
        image_file_name = image_info['file_name']
        # Add to results
        results.append(f"{image_file_name},{caption}")

# Save the results to a text file
with open(output_file_path, 'w') as output_file:
    for line in results:
        output_file.write(line + '\n')

print(f"Results saved to {output_file_path}")


# Paths
# The text file with selected image names
text_file_path = f'{base_dir}/coco.txt'
# Directory where the images are stored
images_directory = '/root/.cache/kagglehub/datasets/nikhil7280/coco-image-caption/versions/1/train2014/train2014'
output_zip_path = f'{base_dir}/cocoimg.zip'   # Path for the output ZIP file

# Read the image file names from the text file
with open(text_file_path, 'r') as file:
    # Use a set to remove duplicates
    image_files = set(line.split(',')[0] for line in file.readlines())

# Compress the selected images into a zip file
with zipfile.ZipFile(output_zip_path, 'w') as zipf:
    for image_file in image_files:
        image_path = os.path.join(images_directory, image_file)
        if os.path.exists(image_path):
            zipf.write(image_path, arcname=image_file)
        else:
            print(f"Image not found: {image_file}")

print(f"Selected images compressed and saved to {output_zip_path}")
