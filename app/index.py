from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import os

app = FastAPI()

# YOUR HF MODEL (NO local files needed!)
HF_TOKEN = os.getenv("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")
tokenizer = AutoTokenizer.from_pretrained("AK47-model-ml/Bert-Sentiment", token=HF_TOKEN)
model = AutoModelForSequenceClassification.from_pretrained("AK47-model-ml/Bert-Sentiment", token=HF_TOKEN)
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

