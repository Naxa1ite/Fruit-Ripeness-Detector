from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import os
from PIL import Image
import io
from tensorflow.keras.models import load_model

app = Flask(__name__, template_folder="templates", static_folder="static")

MODEL_PATH = "best_fruit_model.h5"
IMG_SIZE   = (224, 224)
MAX_BYTES  = 10 * 1024 * 1024

# Alphabetical order from ImageDataGenerator folder names
CLASSES = ["ripe", "rotten", "unripe"]

print(f"Loading model: {MODEL_PATH} ...")
model = load_model(MODEL_PATH)
print(f"Model loaded. Input={model.input_shape}  Output={model.output_shape}")


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def estimate_days(stage: str, probs: list) -> dict:
    p_ripe   = probs[CLASSES.index("ripe")]
    p_rotten = probs[CLASSES.index("rotten")]

    if stage == "unripe":
        days = int((1 - p_ripe) * 5) + 1
        msg  = f"{days} days to become ripe"
        return {"days": days, "message": msg}

    elif stage == "ripe":
        days = int((1 - p_rotten) * 3) + 1
        msg  = f"{days} days to become rotten"
        return {"days": days, "message": msg}

    else:  # rotten
        return {"days": 0, "message": "Already rotten"}


def parse_prediction(probs: np.ndarray) -> dict:
    p          = probs[0].tolist()
    confidence = {CLASSES[i]: round(p[i], 4) for i in range(len(CLASSES))}
    top_idx    = int(np.argmax(p))
    stage      = CLASSES[top_idx]
    days_info  = estimate_days(stage, p)

    return {
        "ripeness":   stage,
        "confidence": confidence,
        "days":       days_info,
    }


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file field 'file' in request."}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    image_bytes = file.read()
    if len(image_bytes) > MAX_BYTES:
        return jsonify({"error": "File too large. Max 10 MB."}), 413

    try:
        x = preprocess(image_bytes)
    except Exception as e:
        return jsonify({"error": f"Cannot process image: {e}"}), 422

    try:
        probs = model.predict(x, verbose=0)
    except Exception as e:
        return jsonify({"error": f"Model inference failed: {e}"}), 500

    return jsonify(parse_prediction(probs))


@app.route("/health")
def health():
    return jsonify({
        "status":  "ok",
        "model":   MODEL_PATH,
        "classes": CLASSES,
        "input":   str(model.input_shape),
        "output":  str(model.output_shape),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
