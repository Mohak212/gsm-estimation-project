from fastapi import FastAPI, UploadFile, File, Form
import shutil
import os

from scripts.predict import predict_gsm

app = FastAPI()

@app.post("/predict")
async def predict(
    cloth_type: str = Form(...),
    image: UploadFile = File(...)
):
    temp_path = "temp_api_image.jpg"

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(image.file, buffer)

    gsm = predict_gsm(temp_path, cloth_type)

    os.remove(temp_path)

    return {
        "cloth_type": cloth_type,
        "predicted_gsm": gsm
    }
