import os
import cv2
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Define 8 super-class mappings matching project architecture
LABEL_MAPPING = {
    'A': 0, 'E': 0, 'M': 0, 'N': 0, 'S': 0, 'T': 0,
    'B': 1, 'D': 1, 'F': 1, 'I': 1, 'U': 1, 'V': 1, 'K': 1, 'R': 1, 'W': 1,
    'C': 2, 'O': 2,
    'G': 3, 'H': 3,
    'L': 4,
    'P': 5, 'Q': 5, 'Z': 5,
    'X': 6,
    'Y': 7, 'J': 7
}

def load_dataset(dataset_dir="AtoZ_3.1", target_size=(400, 400)):
    print(f"Loading dataset from '{dataset_dir}'...")
    X = []
    y = []
    
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory '{dataset_dir}' not found.")
        
    for letter in sorted(os.listdir(dataset_dir)):
        letter_path = os.path.join(dataset_dir, letter)
        if os.path.isdir(letter_path) and letter.upper() in LABEL_MAPPING:
            target_class = LABEL_MAPPING[letter.upper()]
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    if (img.shape[1], img.shape[0]) != target_size:
                        img = cv2.resize(img, target_size)
                    X.append(img)
                    y.append(target_class)
                    
    X = np.array(X, dtype='float32')
    y = np.array(y, dtype='int32')
    print(f"Loaded {len(X)} total samples across 8 super-classes.")
    return X, y

def build_model(input_shape=(400, 400, 3), num_classes=8):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(16, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(96, activation='relu'),
        Dropout(0.5),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    dataset_dir = os.path.join(base_dir, "AtoZ_3.1")
    output_model_path = os.path.join(base_dir, "cnn8grps_rad1_model_new.h5")

    X, y = load_dataset(dataset_dir)
    
    # Shuffle dataset using NumPy
    indices = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    # Train / validation split (80 / 20)
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"Train size: {len(X_train)}, Validation size: {len(X_val)}")
    
    model = build_model()
    model.summary()

    print("Starting training...")
    history = model.fit(X_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_data=(X_val, y_val))

    model.save(output_model_path)
    print(f"Newly trained model saved to '{output_model_path}'")
