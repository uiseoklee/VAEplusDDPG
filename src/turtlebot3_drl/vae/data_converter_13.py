import os
import numpy as np
from PIL import Image as PILImage
import cv2

# Input and output folder paths
input_folders = ["images/depth1/", "images/depth2/"]
output_folders_ori = ["images/converted_depth1/ori_image/", "images/converted_depth2/ori_image/"]
output_folders_npy = ["images/converted_depth1/converted_npy/", "images/converted_depth2/converted_npy/"]
output_folders_png = ["images/converted_depth1/converted_img/", "images/converted_depth2/converted_img/"]

# Additional output folders for modified original images
output_folders_modified_ori_image = ["images/converted_depth1/modified_ori_image/", "images/converted_depth2/modified_ori_image/"]
output_folders_modified_ori_npy = ["images/converted_depth1/modified_ori_npy/", "images/converted_depth2/modified_ori_npy/"]

# Create the output folders if they don't exist
for output_folder in (output_folders_ori + output_folders_npy + output_folders_png +
                      output_folders_modified_ori_image + output_folders_modified_ori_npy):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

# Brightness factors
brightness_factor = 1.0  # 기존 변환 시 밝기 조절을 위한 팩터
darkening_factor = 0.5   # ori_image를 어둡게 조절할 팩터 (0.5는 50% 밝기)

# Define the crop ratios for top and bottom (e.g., 20% from top and 20% from bottom)
crop_ratio_top = 0.0      # 위에서 자를 비율 (0.2는 20%)
crop_ratio_bottom = 0.0   # 아래에서 자를 비율 (0.2는 20%)

# 깊이 값의 최소 및 최대값 (센서의 범위에 따라 조정)
depth_min = 0.05
depth_max = 8.0

# Process each image in both input folders
for input_folder, output_folder_ori, output_folder_npy, output_folder_png in zip(
    input_folders, output_folders_ori, output_folders_npy, output_folders_png):
    
    for filename in os.listdir(input_folder):
        if filename.endswith(".npy"):  # Only process .npy files
            # Load the depth image
            image_path = os.path.join(input_folder, filename)
            try:
                depth_img = np.load(image_path)  # numpy array of shape (H, W) or (H, W, 1)
            except Exception as e:
                print(f"Unable to read .npy file: {image_path}, Error: {e}")
                continue

            # Ensure the depth image has shape (H, W)
            if depth_img.ndim == 3 and depth_img.shape[-1] == 1:
                depth_img = np.squeeze(depth_img, axis=-1)  # (H, W)
            elif depth_img.ndim != 2:
                print(f"Unexpected shape for depth image: {depth_img.shape} in file {image_path}")
                continue

            # Save the original depth image as PNG for visualization
            # Scale depth values to 0-255 for visualization
            depth_scaled = np.clip((depth_img - depth_min) / (depth_max - depth_min), 0, 1)
            depth_scaled = (depth_scaled * 255).astype(np.uint8)
            ori_image = PILImage.fromarray(depth_scaled)
            ori_output_path = os.path.join(output_folder_ori, filename.replace('.npy', '.png'))
            ori_image.save(ori_output_path)
            print(f"Saved original depth image as PNG: {ori_output_path}")

            # Get the height and width of the image
            height, width = depth_img.shape

            # Calculate the number of pixels to crop from top and bottom
            crop_pixels_top = int(height * crop_ratio_top)
            crop_pixels_bottom = int(height * crop_ratio_bottom)

            # Define the start and end points for cropping
            start_row = crop_pixels_top
            end_row = height - crop_pixels_bottom

            # Ensure that start_row is less than end_row
            if start_row >= end_row:
                print(f"Crop ratios too high for image: {image_path}")
                continue

            # Crop the image to retain only the central part
            cropped_img = depth_img[start_row:end_row, :]

            # Get the new height after cropping
            new_height = end_row - start_row

            # Calculate the scaling factor to adjust width based on height reduction
            # This ensures the image becomes wider proportionally
            scaling_factor = height / new_height  # 예: height가 100이고 new_height가 60이면 scaling_factor는 약 1.666

            # Calculate the new width to make the image wider
            new_width = int(width * scaling_factor)

            # Resize the cropped image to make it wider
            resized_img = cv2.resize(cropped_img, (new_width, height), interpolation=cv2.INTER_LINEAR)

            # If the new width is greater than original, you can choose to:
            # a) Crop the width to original
            # b) Pad the width to original
            # c) Keep the resized image as is
            # Here, we'll pad the image to match the original width

            if new_width > width:
                # Calculate the amount to crop from each side
                excess_width = new_width - width
                crop_left = excess_width // 2
                crop_right = excess_width - crop_left
                final_img = resized_img[:, crop_left:new_width - crop_right]
            elif new_width < width:
                # Calculate the amount to pad on each side
                deficit_width = width - new_width
                pad_left = deficit_width // 2
                pad_right = deficit_width - pad_left
                final_img = cv2.copyMakeBorder(
                    resized_img,
                    top=0,
                    bottom=0,
                    left=pad_left,
                    right=pad_right,
                    borderType=cv2.BORDER_CONSTANT,
                    value=depth_min  # 깊이 값의 최소값으로 패딩 (0.05)
                )
            else:
                final_img = resized_img  # Width matches original

            # Brighten the image by multiplying the pixel values by the brightness factor
            # Since depth values are float32, ensure they remain within the valid range
            img_bright = np.clip(final_img * brightness_factor, depth_min, depth_max).astype(np.float32)

            # Save the modified depth image as .npy
            converted_npy_path = os.path.join(output_folder_npy, filename)
            try:
                np.save(converted_npy_path, img_bright)
                print(f"Saved converted depth image as .npy: {converted_npy_path}")
            except Exception as e:
                print(f"Failed to save converted .npy file: {converted_npy_path}, Error: {e}")
                continue

            # Additionally, save the converted depth image as .png for visualization
            # Scale depth values to 0-255
            img_bright_scaled = np.clip((img_bright - depth_min) / (depth_max - depth_min), 0, 1)
            img_bright_scaled = (img_bright_scaled * 255).astype(np.uint8)
            converted_png = PILImage.fromarray(img_bright_scaled)
            converted_png_path = os.path.join(output_folder_png, filename.replace('.npy', '.png'))
            try:
                converted_png.save(converted_png_path)
                print(f"Saved converted depth image as PNG: {converted_png_path}")
            except Exception as e:
                print(f"Failed to save converted PNG file: {converted_png_path}, Error: {e}")
                continue

            print(f"Processed and saved image: {filename}\n")

print("All .npy depth images have been processed and saved.")

# --- 추가: ori_image 폴더 내 이미지들을 어둡게 조정하여 modified_ori_image와 modified_ori_npy에 저장 ---

# Output folders for modified original images
output_folders_modified_ori_image = ["images/converted_depth1/modified_ori_image/", "images/converted_depth2/modified_ori_image/"]
output_folders_modified_ori_npy = ["images/converted_depth1/modified_ori_npy/", "images/converted_depth2/modified_ori_npy/"]

# Define darkening factor (e.g., 0.5 for 50% brightness)
darkening_factor = 0.5  # 조명을 어둡게 할 정도를 조절 (0.0 ~ 1.0)

# Process each ori_image to create modified_ori_image and modified_ori_npy
for output_folder_ori, output_folder_modified_ori_image, output_folder_modified_ori_npy in zip(
    output_folders_ori, output_folders_modified_ori_image, output_folders_modified_ori_npy):
    
    for filename in os.listdir(output_folder_ori):
        if filename.endswith(".png"):  # Only process .png files
            # Load the original depth image as PNG
            ori_image_path = os.path.join(output_folder_ori, filename)
            try:
                ori_img = PILImage.open(ori_image_path).convert('L')  # Ensure it's in grayscale
                ori_img_np = np.array(ori_img).astype(np.float32)  # (H, W) with values 0-255
            except Exception as e:
                print(f"Unable to read original PNG file: {ori_image_path}, Error: {e}")
                continue

            # Invert scaling to get back to depth values
            depth_img = (ori_img_np / 255.0) * (depth_max - depth_min) + depth_min  # (H, W) with depth_min - depth_max

            # Apply darkening
            depth_dark = np.clip(depth_img * darkening_factor, depth_min, depth_max).astype(np.float32)

            # Save the modified depth image as .npy
            modified_npy_filename = filename.replace('.png', '.npy')
            modified_npy_path = os.path.join(output_folder_modified_ori_npy, modified_npy_filename)
            try:
                np.save(modified_npy_path, depth_dark)
                print(f"Saved modified original depth image as .npy: {modified_npy_path}")
            except Exception as e:
                print(f"Failed to save modified original .npy file: {modified_npy_path}, Error: {e}")
                continue

            # Scale depth_dark to 0-255 for visualization
            depth_dark_scaled = np.clip((depth_dark - depth_min) / (depth_max - depth_min), 0, 1)
            depth_dark_scaled = (depth_dark_scaled * 255).astype(np.uint8)
            modified_png = PILImage.fromarray(depth_dark_scaled)
            modified_png_filename = filename.replace('.png', '.png')  # Keep the same filename
            modified_png_path = os.path.join(output_folder_modified_ori_image, modified_png_filename)
            try:
                modified_png.save(modified_png_path)
                print(f"Saved modified original depth image as PNG: {modified_png_path}")
            except Exception as e:
                print(f"Failed to save modified original PNG file: {modified_png_path}, Error: {e}")
                continue

            print(f"Modified and saved original image: {filename}\n")

print("All original depth images have been modified and saved.")
