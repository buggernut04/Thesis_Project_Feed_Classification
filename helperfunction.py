import numpy as np 
from matplotlib import pyplot as plt
import os
import cv2
import Augmentor
import random
from PIL import Image
from collections import Counter
import shutil

def resize_and_crop_images(directory_path, output_path, target_size=(400, 400)):
    if not os.path.isdir(directory_path):
        print(f"Error: '{directory_path}' is not a valid directory.")
        return

    for filename in os.listdir(directory_path):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):  # Adjust file extensions as needed
            image_path = os.path.join(directory_path, filename)
            try:
                with Image.open(image_path) as img:
                    # # Do something with the image, e.g., display, process, or save
                    # print(f"Processing image: {image_path}")
                    # # Example: Resize the image
                    # resized_img = img.resize((256, 256))
                    # resized_img.save(f"resized_{filename}")

                    width, height = img.size

                    # Calculate the crop box to center the image within the target size
                    left = (width - target_size[0]) / 2
                    top = (height - target_size[0]) / 2
                    right = (width + target_size[0]) / 2
                    bottom = (height + target_size[0]) / 2

                    # Crop the image   
                    cropped_img = img.crop((left, top, right, bottom))

                    # Resize the cropped image to the target size
                    # resized_img = img.resize(target_size) 

                    # Create the output directory if it doesn't exist
                    os.makedirs(output_path, exist_ok=True)

                    # Save the cropped image
                    output_file = os.path.join(output_path, os.path.basename(image_path))
                    # resized_img.save(output_file)
                    cropped_img.save(output_file)
            except Exception as e:
                print(f"Error processing image '{image_path}': {e}")

def resize_and_subtract_mean(img_samp, target_size):
    # ImageNet mean RGB values (you can adjust these if needed)
    MEAN_RGB = [123.68, 116.779, 103.939]

    # Convert BGR to RGB for Matplotlib display
    img_rgb = cv2.cvtColor(img_samp, cv2.COLOR_BGR2RGB)

    #print(img_rgb.shape)

    t_size = target_size
    pos = 2

    # Define the rectangle coordinates
    x1 = (img_rgb.shape[1] - t_size) // pos
    y1 = (img_rgb.shape[0] - t_size) // pos
    x2 = (img_rgb.shape[1] + t_size) // pos
    y2 = (img_rgb.shape[0] + t_size) // pos

    # Extract the region of interest (ROI) within the rectangle
    roi = img_rgb[y1:y2, x1:x2]

    # Resize the ROI to 400x400
    resized_roi = cv2.resize(roi, (400, 400))

    # Subtract mean RGB from each channel
    resized_roi = resized_roi.astype(np.float32)  # Important: Convert to float before subtracting!
    resized_roi[:, :, 0] -= MEAN_RGB[0]  # Subtract from Red channel
    resized_roi[:, :, 1] -= MEAN_RGB[1]  # Subtract from Green channel
    resized_roi[:, :, 2] -= MEAN_RGB[2]  # Subtract from Blue channel

    return resized_roi


def segment_img(gray_img):
    # Compute histogram
    hist, _ = np.histogram(gray_img.ravel(), bins=256, range=[0, 256])

    # Total number of pixels
    total_pixels = gray_img.size

    # Initialize variables
    maximum_variance = 0.0
    optimal_threshold = 0

    for threshold in range(256):
        w_background = np.sum(hist[:threshold]) / total_pixels
        w_foreground = 1.0 - w_background

        if w_background == 0 or w_foreground == 0:
            continue  # Skip invalid cases

        mu_background = np.sum(np.arange(threshold) * hist[:threshold]) / (w_background * total_pixels)
        mu_foreground = np.sum(np.arange(threshold, 256) * hist[threshold:]) / (w_foreground * total_pixels)

        intra_class_variance = w_background * w_foreground * (mu_background - mu_foreground) ** 2

        if intra_class_variance > maximum_variance:
            maximum_variance = intra_class_variance
            optimal_threshold = threshold

    # Apply Otsu's thresholding
    _, thresh_img = cv2.threshold(gray_img, optimal_threshold, 255, cv2.THRESH_BINARY)

    return thresh_img

def extract_and_resize_foreground(img, opened_img, target_size=(400, 400)):
    
    # Apply threshold to separate foreground from background
    # Since the background is dark and foreground is light,
    # we can use THRESH_BINARY + THRESH_OTSU for automatic thresholding
    _, thresh = cv2.threshold(opened_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a mask
    mask = np.zeros_like(opened_img)
    
    # Draw the largest contour on the mask
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest_contour], -1, (255, 255, 255), -1)
    
    # Apply the mask to the original image
    result = cv2.bitwise_and(img, img, mask=mask)
    
    # Get the bounding box of the foreground
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # Crop to the bounding box
    cropped = result[y:y+h, x:x+w]
    
    # Resize to target size
    resized = cv2.resize(cropped, target_size, interpolation=cv2.INTER_AREA)
    
    # Create a black background (changed from white to black)
    black_bg = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
    
    # Create a mask for the resized image
    gray_resized = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, mask_resized = cv2.threshold(gray_resized, 1, 255, cv2.THRESH_BINARY)
    
    # Blend the resized image with the black background
    mask_resized_3ch = cv2.cvtColor(mask_resized, cv2.COLOR_GRAY2BGR)
    final_result = np.where(mask_resized_3ch == 0, black_bg, resized)
    
    return final_result

def transfer_image_train_and_test(input_path, testing_path, training_path):
    # Get a list of files in the directory, sorted by filename
    files = sorted(os.listdir(input_path), key=lambda x: int(x.split('(')[1].split(')')[0]))
    index = 1

    for filename in files:
        filepath = os.path.join(input_path, filename)
        # print(filepath)
        # print(type(filepath))
        # print(type(filepath))
            
        if os.path.isfile(filepath):
            destination_path = os.path.join(testing_path, filename) if index <= 12 else os.path.join(training_path, filename)

            # Move the file using shutil.move (handles cases where source and dest are on different drives)
            shutil.copy2(filepath, destination_path)  # or shutil.copy2(src, dst) to copy and keep original
            print(f"Image moved successfully from {input_path} to {destination_path}")

        index += 1

        if index > 20:
            index = 1

def img_augmentation(input_folder, data_size):
    p = Augmentor.Pipeline(input_folder)
    
    # scaling
    p.zoom(probability=0.5, min_factor=0.5, max_factor=1.5)

    # rotation / flip
    p.flip_top_bottom(probability=0.5)
    p.flip_left_right(probability=0.5)

    # shifting
    p.random_distortion(probability=0.5, grid_width=4, grid_height=4, magnitude=8)

    # lighting
    p.random_brightness(probability=0.5, min_factor=0.8, max_factor=1.2)
    p.random_contrast(probability=0.5, min_factor=0.8, max_factor=1.2)

    # number of samples
    p.sample(data_size)

