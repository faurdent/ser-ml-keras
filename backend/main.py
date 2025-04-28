from fastapi import FastAPI, Request, Form, UploadFile
from typing import Annotated
from fastapi.middleware.cors import CORSMiddleware
from classifier import SpeechEmotionClassifier, model


app = FastAPI()

ser_classifier = SpeechEmotionClassifier()

origins = [
    "http://localhost:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/emotion-recognition")
async def file(request: Request, file: Annotated[UploadFile, Form()]):
    emotion, probability = ser_classifier.predict_emotion(file.file)
    return {"label": emotion, "probability": probability}
