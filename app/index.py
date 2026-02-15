from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = FastAPI()

# YOUR HF MODEL (NO local files needed!)
model = AutoModelForSequenceClassification.from_pretrained("AK47-model-ml")
tokenizer = AutoTokenizer.from_pretrained("AK47-model-ml")
model.eval()

class PredictionRequest(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "BERT Sentiment Pro API (96.5% F1) - LIVE!"}

@app.post("/predict")
async def predict(request: PredictionRequest):
    inputs = tokenizer(request.text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=-1).item()
    return {
        "sentiment": "POSITIVE" if pred == 1 else "NEGATIVE",
        "confidence": float(torch.max(probs, dim=-1)[0])
    }

