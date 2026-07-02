from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
import json
import cv2
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

IMG_SIZE = 64

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
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    predictions = model.predict(img)
    class_idx = str(np.argmax(predictions[0]))
    confidence = float(np.max(predictions[0]))
    english_text = reverse_map.get(class_idx, "Unknown")
    clean_text = english_text.replace("ASL_", "").replace("ISL_", "")

    try:
        translated = GoogleTranslator(
            source='en',
            target=target_lang
        ).translate(clean_text)
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