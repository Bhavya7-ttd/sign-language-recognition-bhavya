import os
import sys
import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

def p(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()

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

CLASS_NAMES = ["Class 0 [aemnst]", "Class 1 [bdfiuvkrw]", "Class 2 [co]",
               "Class 3 [gh]", "Class 4 [l]", "Class 5 [pqz]",
               "Class 6 [x]", "Class 7 [yj]"]

def evaluate_model(model_path="cnn8grps_rad1_model.h5", dataset_dir="AtoZ_3.1"):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_model_path = os.path.join(base_dir, model_path)
    full_dataset_dir = os.path.join(base_dir, dataset_dir)

    p(f"Evaluating model '{full_model_path}' on dataset '{full_dataset_dir}'...")

    if not os.path.exists(full_model_path):
        p(f"Error: Model file '{full_model_path}' not found.")
        return

    if not os.path.exists(full_dataset_dir):
        p(f"Error: Dataset directory '{full_dataset_dir}' not found.")
        return

    model = load_model(full_model_path)
    
    total_correct = 0
    total_samples = 0
    class_correct = [0] * 8
    class_totals = [0] * 8

    for letter in sorted(os.listdir(full_dataset_dir)):
        letter_path = os.path.join(full_dataset_dir, letter)
        if os.path.isdir(letter_path) and letter.upper() in LABEL_MAPPING:
            target_class = LABEL_MAPPING[letter.upper()]
            imgs = []
            for img_name in os.listdir(letter_path):
                img_path = os.path.join(letter_path, img_name)
                img = cv2.imread(img_path)
                if img is not None:
                    if (img.shape[1], img.shape[0]) != (400, 400):
                        img = cv2.resize(img, (400, 400))
                    imgs.append(img)
            
            if imgs:
                batch_x = np.array(imgs, dtype='float32')
                preds = model.predict(batch_x, batch_size=32, verbose=0)
                pred_classes = np.argmax(preds, axis=1)
                
                correct = int(np.sum(pred_classes == target_class))
                total = len(imgs)
                
                total_correct += correct
                total_samples += total
                class_correct[target_class] += correct
                class_totals[target_class] += total
                p(f"Processed letter '{letter}' ({total} images) -> Correct: {correct}/{total}")

    overall_acc = (total_correct / max(total_samples, 1)) * 100.0
    p("\n" + "="*50)
    p(f"EVALUATION RESULTS FOR '{model_path}'")
    p(f"Total Samples Evaluated: {total_samples}")
    p(f"Overall Super-Class Accuracy: {overall_acc:.2f}% ({total_correct}/{total_samples})")
    p("="*50)

    p("\nPER-CLASS ACCURACY BREAKDOWN:")
    for cls_idx in range(8):
        cnt = class_totals[cls_idx]
        corr = class_correct[cls_idx]
        acc = (corr / cnt * 100.0) if cnt > 0 else 0.0
        p(f"  {CLASS_NAMES[cls_idx]:<25}: {acc:.2f}% ({corr}/{cnt})")

if __name__ == "__main__":
    evaluate_model()
