import json
import numpy as np
from PIL import Image
import tensorflow as tf

MODEL_PATH = 'models/skin_model'
LABELS_PATH = 'models/skin_labels.json'
IMG_PATH = 'datasets/skin/Lupus/lupus-acute-21.jpeg'

print('Loading model...')
model = tf.keras.models.load_model(MODEL_PATH)
print('Model loaded from', MODEL_PATH)

with open(LABELS_PATH) as f:
    labels = json.load(f)
print('Labels:', labels)

img = Image.open(IMG_PATH).convert('RGB').resize((224,224))
arr = np.array(img)/255.0
arr = arr.reshape(1,224,224,3)

pred = model.predict(arr)
probs = pred[0]
idx = int(np.argmax(probs))
confidence = float(probs[idx])
class_name = labels[idx] if labels else str(idx)

# top 3
top3_idx = probs.argsort()[-3:][::-1]
print('\nPrediction:')
print(json.dumps({
    'image': IMG_PATH,
    'predicted_class': class_name.lower(),
    'confidence': confidence,
    'top3': [
        {'class': labels[i].lower() if labels else str(i), 'prob': float(probs[i])}
        for i in top3_idx
    ]
}, indent=2))
