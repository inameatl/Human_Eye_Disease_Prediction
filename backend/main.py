from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
from tensorflow.keras.utils import load_img, img_to_array
import tempfile
from recommendation import cnv, dme, drusen, normal

# Initialize FastAPI app
app = FastAPI()

# Allow CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once at startup
model = tf.keras.models.load_model(r"C:\xray disease\Expariments\Human_Eye_Disease_Prediction\model\Trained_Model.keras")
class_names = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name

        img = load_img(tmp_path, target_size=(224, 224))
        x = img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        predictions = model.predict(x)[0]  # single sample
        idx = int(np.argmax(predictions))
        disease = class_names[idx]

        rec_map = {
            "CNV": cnv,
            "DME": dme,
            "DRUSEN": drusen,
            "NORMAL": normal
        }

        probs = [
            {"class": cls, "confidence": float(p) * 100}
            for cls, p in zip(class_names, predictions)
        ]

        return {
            "prediction": disease,
            "recommendation": rec_map[disease],
            "probabilities": probs
        }
    except Exception as e:
        return {"error": str(e)}
