from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from pathlib import Path
import torch
from transformers import AutoTokenizer ,AutoModelForSequenceClassification

app = FastAPI()

MODEL_DIR = Path("models/distilbert/500k")

tokenizer = None
model = None
device = "cpu"

class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)

@app.on_event("startup")
def load_model():
    global tokenizer, model, device

    if not MODEL_DIR.exists():
        raise RuntimeError(f"Model folder not found: {MODEL_DIR.resolve()}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
    model.to(device)
    model.eval()

    print(f"Model loaded on device: {device}")

@app.get("/")
def root():
    return {"message": "Sentiment API is running"}

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None and tokenizer is not None,
        "device": device,
        "model_dir": str(MODEL_DIR)
    }

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    text = req.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text is empty")
    
    inputs = tokenizer(
        text,
        truncation=True,
        padding=True,
        max_length=256,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

    pred_id = int(torch.argmax(probs).item())
    score = float(probs[pred_id].item())

    id2label = getattr(model.config, "id2label", None)
    if id2label:
        label = id2label[pred_id]
    else:
        label = f"LABEL_{pred_id}"

    if label is not None:
        label = "POSITIVE" if pred_id == 1 else "NEGATIVE"
    
    probs_dict = {
        "NEGATIVE": float(probs[0].item()),
        "POSITIVE": float(probs[1].item())
    }
    return{
        "text": text,
        "label": label,
        "score": score,
        "probs": probs_dict,
        "device": device
    }