import numpy as np
import cv2
import shap
from ultralytics import YOLO
from skimage.transform import resize
import matplotlib.pyplot as plt

# Load model and classes
model = YOLO('runs/detect/train/weights/best.pt') 
class_names = ['Aadhar', 'pan-card']

def predict_fn(images):
    results_probs = []
    for img in images:
        img_uint8 = img.astype(np.uint8)
        results = model(img_uint8, verbose=False)

        probs = np.zeros(len(class_names))
        boxes = results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                probs[cls_id] += conf

        # probs = probs + 1e-6
        # probs /= probs.sum()
        results_probs.append(probs)
    return np.array(results_probs)

# Load and preprocess image
image_path = "images/ad1.jpg"
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# img_resized = resize(img, (224, 224), preserve_range=True).astype(np.uint8)
img_resized = img.astype(np.uint8)
X = np.expand_dims(img_resized, axis=0)  # shape (1,H,W,C)

probs = predict_fn([img_resized])[0]  # single image
predicted_class_id = np.argmax(probs)
predicted_class = class_names[predicted_class_id]
confidence = probs[predicted_class_id]

print(f"Model Prediction: {predicted_class} (Confidence: {confidence:.2f})")
print("-------------------------------\n",probs)

masker = shap.maskers.Image("inpaint_telea", X[0].shape)
explainer = shap.Explainer(predict_fn, masker, output_names=class_names)

shap_values = explainer(X, max_evals=100, batch_size=1, outputs=shap.Explanation.argsort.flip[:1])

X_norm = X.astype(np.float32) / 255.0  # normalize to 0-1

shap.image_plot(shap_values.values, X_norm)

