from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import cv2
import mediapipe as mp
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import io
import os
import gdown

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Download model from Google Drive if not present
MODEL_PATH = "signsetu_best.h5"
FILE_ID = "1PYSymFsN3HQd4qr_Ob1Loj-RVgGysV8s"

if not os.path.exists(MODEL_PATH):
    print("Downloading model from Google Drive...")
    gdown.download(f"https://drive.google.com/uc?id={FILE_ID}", MODEL_PATH, quiet=False)
    print("Model downloaded ✅")

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
with open('reverse_label_map.json') as f:
    reverse_map = json.load(f)
print("Model loaded ✅")

# MediaPipe hands — runs once at startup, reused for every request
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print("MediaPipe loaded ✅")

IMG_SIZE = 64
PADDING = 0.20   # 20% padding around the hand bounding box


def crop_hand(img_rgb):
    """
    Detect hand landmarks, compute bounding box, return cropped hand image.
    Returns None if no hand found.
    """
    h, w = img_rgb.shape[:2]
    results = hands_detector.process(img_rgb)

    if not results.multi_hand_landmarks:
        return None

    lm = results.multi_hand_landmarks[0].landmark

    xs = [p.x for p in lm]
    ys = [p.y for p in lm]

    x_min = max(0, int((min(xs) - PADDING) * w))
    x_max = min(w, int((max(xs) + PADDING) * w))
    y_min = max(0, int((min(ys) - PADDING) * h))
    y_max = min(h, int((max(ys) + PADDING) * h))

    cropped = img_rgb[y_min:y_max, x_min:x_max]
    if cropped.size == 0:
        return None
    return cropped


@app.get("/")
def home():
    return {"status": "SignSetu API is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    target_lang: str = "ta"
):
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # --- Crop hand region ---
    hand_crop = crop_hand(img_rgb)

    if hand_crop is None:
        return {
            "english": "Unknown",
            "translated": "",
            "audio": "",
            "confidence": 0.0,
            "language": target_lang
        }

    # --- Resize cropped hand to model input size ---
    hand_resized = cv2.resize(hand_crop, (IMG_SIZE, IMG_SIZE))
    hand_input = hand_resized.astype(np.float32) / 255.0
    hand_input = np.expand_dims(hand_input, axis=0)

    # --- Predict ---
    predictions = model.predict(hand_input)
    class_idx = str(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    english_text = reverse_map.get(class_idx, "Unknown")
    clean_text = english_text.replace("ASL_", "").replace("ISL_", "")

    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(clean_text)
    except:
        translated = clean_text

    try:
        tts = gTTS(text=translated, lang=target_lang, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    except:
        audio_b64 = ""

    return {
        "english": clean_text,
        "translated": translated,
        "audio": audio_b64,
        "confidence": round(confidence * 100, 2),
        "language": target_lang
    }