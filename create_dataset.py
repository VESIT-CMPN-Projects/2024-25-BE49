import os
import pickle
import time
import cv2
import mediapipe as mp
from tqdm import tqdm

def create_dataset():
    # Initialize MediaPipe Hands with optimized parameters
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,  # Detect up to 2 hands
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    DATA_DIR = 'F:\\SilentCue - Hand Gesture Recognition for Deaf and Non Verbal\\data'
    output_file = 'data.pickle'
    
    data = []
    labels = []
    skipped_images = 0
    successful_images = 0

    # First count total images for accurate progress tracking
    print("Counting images...")
    image_paths = []
    for dir_ in os.listdir(DATA_DIR):
        folder_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(folder_path):
            for img_path in os.listdir(folder_path):
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append((folder_path, img_path, dir_))
    
    total_images = len(image_paths)
    print(f"Found {total_images} images across {len(set(p[2] for p in image_paths))} classes")

    start_time = time.time()
    
    # Process images with progress bar
    for folder_path, img_path, dir_ in tqdm(image_paths, desc="Processing images"):
        try:
            img = cv2.imread(os.path.join(folder_path, img_path))
            if img is None:
                skipped_images += 1
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if not results.multi_hand_landmarks:
                skipped_images += 1
                continue

            data_aux = []
            x_, y_ = [], []

            # Process all detected hands (up to 2)
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    x_.append(lm.x)
                    y_.append(lm.y)

                # Normalize landmarks relative to hand position
                min_x, min_y = min(x_), min(y_)
                for lm in hand_landmarks.landmark:
                    data_aux.extend([lm.x - min_x, lm.y - min_y])

            if data_aux:  # Only add if we got landmarks
                data.append(data_aux)
                labels.append(dir_)
                successful_images += 1

        except Exception as e:
            print(f"\nError processing {img_path}: {str(e)}")
            skipped_images += 1

    hands.close()  # Important: release MediaPipe resources

    # Save dataset with compression
    print("\nSaving dataset...")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'data': data,
            'labels': labels,
            'metadata': {
                'total_images': total_images,
                'successful_images': successful_images,
                'skipped_images': skipped_images,
                'processing_time': time.time() - start_time
            }
        }, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Print summary
    print(f"\nDataset creation completed!")
    print(f"Successfully processed: {successful_images}/{total_images} ({successful_images/total_images:.1%})")
    print(f"Skipped images: {skipped_images} (no hands detected or errors)")
    print(f"Processing time: {time.time() - start_time:.2f} seconds")
    print(f"Saved to: {output_file}")

if __name__ == "__main__":
    create_dataset()