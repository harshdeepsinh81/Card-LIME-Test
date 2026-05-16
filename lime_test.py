import numpy as np
import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO
from lime import lime_image
from skimage.segmentation import mark_boundaries

model=YOLO('runs/detect/train/weights/best.pt')

class_names=['Aadhar', 'pan-card']

def predict_fn(images):
    results_probs=[]

    for img in images:
        img_unit8 = (img).astype(np.uint8)

        results=model(img_unit8)

        probs=np.zeros(len(class_names))

        boxes=results[0].boxes

        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                cls_id=int(box.cls[0].item())
                conf=float(box.conf[0].item())
                
                probs[cls_id]=max(probs[cls_id], conf)

        if probs.sum() > 0:
            probs /= probs.sum()
        else:
            probs=np.ones(len(class_names)) / len(class_names)

        results_probs.append(probs)

    return np.array(results_probs)

image=cv2.imread('images/test.jpg')
image=cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

probs = predict_fn([image])[0] 
predicted_class = class_names[np.argmax(probs)]
confidence = probs[np.argmax(probs)]
print(f"Model Prediction: {predicted_class} (Confidence: {confidence:.2f})")

explainer=lime_image.LimeImageExplainer()

explanation=explainer.explain_instance(
    image=image, 
    classifier_fn=predict_fn, 
    top_labels=1, 
    hide_color=0, 
    num_samples=300)

temp,mask=explanation.get_image_and_mask(
    label=explanation.top_labels[0], 
    positive_only=True,
    num_features=5, 
    hide_rest=False)

# Mark boundaries for LIME
lime_image_marked = mark_boundaries(temp / 255.0, mask)

# Convert to uint8 for OpenCV text overlay
lime_image_uint8 = (lime_image_marked * 255).astype(np.uint8)

plt.figure(figsize=(8, 8))
plt.imshow(lime_image_uint8)
plt.axis('off')
plt.title(f"Prediction: {predicted_class} (Confidence: {confidence:.2f})", fontsize=16, color='red')
plt.show()