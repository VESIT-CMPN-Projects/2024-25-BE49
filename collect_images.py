import cv2
import os
import numpy as np
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Define the base path to the folders containing images
base_path = r'F:\SilentCue - Hand Gesture Recognition for Deaf and Non Verbal\data'  # Update this path
output_base_path = r'F:\SilentCue - Hand Gesture Recognition for Deaf and Non Verbal\processed_and_extracted_data'  # Base output folder

# Ensure the output base directory exists
os.makedirs(output_base_path, exist_ok=True)

# Function to process images
def process_images(folder_path, output_folder):
    image_count = 1  # Initialize image counter for renaming

    for filename in os.listdir(folder_path):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Process only image files
            img_path = os.path.join(folder_path, filename)
            image = cv2.imread(img_path)

            if image is None:
                continue

            # Convert to RGB for MediaPipe
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_image)

            # Check if any hand is detected
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Create a mask for the detected hand
                    mask = np.zeros(image.shape[:2], dtype=np.uint8)
                    for landmark in hand_landmarks.landmark:
                        x = int(landmark.x * image.shape[1])
                        y = int(landmark.y * image.shape[0])
                        cv2.circle(mask, (x, y), 5, (255), -1)

                    # Apply Gaussian Blur to the original image
                    blurred = cv2.GaussianBlur(image, (15, 15), 0)

                    # Combine the blurred image with the mask
                    blurred_roi = cv2.bitwise_and(blurred, blurred, mask=mask)

                    # Convert to grayscale
                    gray_blurred_roi = cv2.cvtColor(blurred_roi, cv2.COLOR_BGR2GRAY)

                    # Create output filename with sequential numbering
                    output_filename = f"{image_count}.jpg"
                    cv2.imwrite(os.path.join(output_folder, output_filename), gray_blurred_roi)

                    # Increment the image counter
                    image_count += 1

# Process each folder (0-25)
for i in range(26):  # Folder names are numbered from 0 to 25
    folder_path = os.path.join(base_path, str(i))  # Input folder path
    output_folder = os.path.join(output_base_path, str(i))  # Output folder path

    # Ensure the output folder for each input folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Process images in the current folder and save to the corresponding output folder
    process_images(folder_path, output_folder)
