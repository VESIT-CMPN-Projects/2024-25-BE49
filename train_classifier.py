import pickle
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm

def train_classifier():
    # Load dataset
    try:
        with open('./data.pickle', 'rb') as f:
            data_dict = pickle.load(f)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    # Verify data structure
    if 'data' not in data_dict or 'labels' not in data_dict:
        print("Invalid data format: expected 'data' and 'labels' keys")
        return

    # Filter and validate data
    valid_indices = [i for i, item in enumerate(data_dict['data']) if len(item) == 42]
    data = np.array([data_dict['data'][i] for i in valid_indices])
    labels = np.array([data_dict['labels'][i] for i in valid_indices])

    if len(data) == 0:
        print("Error: No valid samples with 42 features (21 landmarks * 2 coordinates)")
        return

    print(f"\nDataset loaded: {len(data)} samples, {len(np.unique(labels))} classes")

    # Split dataset (stratified)
    x_train, x_test, y_train, y_test = train_test_split(
        data, labels, 
        test_size=0.2, 
        shuffle=True, 
        stratify=labels,
        random_state=42
    )

    # Initialize model with better parameters
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        n_jobs=-1,  # Use all CPU cores
        random_state=42
    )

    print("\nTraining model...")
    start_time = time.time()
    
    # Train the model (proper batch training)
    model.fit(x_train, y_train)
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")

    # Evaluate
    print("\nEvaluating model...")
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"\nOverall Accuracy: {accuracy * 100:.2f}%")

    # Save model with metadata
    model_data = {
        'model': model,
        'accuracy': accuracy,
        'classes': np.unique(labels).tolist(),
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'input_shape': data[0].shape
    }

    with open('model.p', 'wb') as f:
        pickle.dump(model_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print("\nModel saved to 'model.p'")

if __name__ == "__main__":
    train_classifier()