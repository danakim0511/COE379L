from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from io import BytesIO
from PIL import Image

app = FastAPI()

model = load_model("model/best_model.h5")

@app.get("/summary")
def summary():
    return {
        "model_name": "Alternate LeNet-5 CNN",
        "input_shape": [128, 128, 3],
        "output": ["damage", "no_damage"]
    }

@app.post("/inference")
async def inference(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(BytesIO(contents)).resize((128, 128))
    image = img_to_array(image) / 255.0
    image = np.expand_dims(image, axis=0)

    pred = model.predict(image)[0][0]
    label = "damage" if pred >= 0.5 else "no_damage"
    return JSONResponse(content={"prediction": label})
