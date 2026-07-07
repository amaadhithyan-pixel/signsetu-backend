from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import cv2
import mediapipe as mp
from PIL import Image
import torch
from transformers import AutoImageProcessor, SiglipForImageClassification
from deep_translator import GoogleTranslator
from gtts import gTTS
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load Hugging Face model ──────────────────────────────────────────
MODEL_NAME = "prithivMLmods/Alphabet-Sign-Language-Detection"
print("Loading ASL model from Hugging Face...")
processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
model     = SiglipForImageClassification.from_pretrained(MODEL_NAME)
model.eval()
print("Model loaded ✅")

LABELS = {
    "0":"A","1":"B","2":"C","3":"D","4":"E","5":"F","6":"G","7":"H",
    "8":"I","9":"J","10":"K","11":"L","12":"M","13":"N","14":"O","15":"P",
    "16":"Q","17":"R","18":"S","19":"T","20":"U","21":"V","22":"W","23":"X",
    "24":"Y","25":"Z"
}

# ── MediaPipe hand crop ──────────────────────────────────────────────
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)
print("MediaPipe loaded ✅")

PADDING = 0.20

def crop_hand(img_rgb):
    h, w = img_rgb.shape[:2]
    res = hands_detector.process(img_rgb)
    if not res.multi_hand_landmarks:
        return None
    lm = res.multi_hand_landmarks[0].landmark
    xs = [p.x for p in lm]; ys = [p.y for p in lm]
    x1 = max(0, int((min(xs) - PADDING) * w))
    x2 = min(w, int((max(xs) + PADDING) * w))
    y1 = max(0, int((min(ys) - PADDING) * h))
    y2 = min(h, int((max(ys) + PADDING) * h))
    crop = img_rgb[y1:y2, x1:x2]
    return crop if crop.size > 0 else None


@app.get("/")
def home():
    return {"status": "SignSetu API is running"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    target_lang: str = "ta"
):
    contents = await file.read()
    nparr   = np.frombuffer(contents, np.uint8)
    img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Crop hand
    hand_crop = crop_hand(img_rgb)
    if hand_crop is None:
        return {"english": "Unknown", "translated": "", "audio": "", "confidence": 0.0, "language": target_lang}

    # Run HuggingFace model
    pil_img = Image.fromarray(hand_crop)
    inputs  = processor(images=pil_img, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    probs      = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    class_idx  = int(torch.argmax(logits, dim=1).item())
    confidence = float(probs[class_idx])
    letter     = LABELS.get(str(class_idx), "Unknown")

    try:
        translated = GoogleTranslator(source='en', target=target_lang).translate(letter)
    except:
        translated = letter

    try:
        tts    = gTTS(text=translated, lang=target_lang, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        audio_b64 = base64.b64encode(buffer.read()).decode('utf-8')
    except:
        audio_b64 = ""

    return {
        "english":    letter,
        "translated": translated,
        "audio":      audio_b64,
        "confidence": round(confidence * 100, 2),
        "language":   target_lang
    }