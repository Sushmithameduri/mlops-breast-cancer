from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.predict import predict

app = FastAPI()


class InputData(BaseModel):
    features: list

@app.post("/predict")
def get_prediction(data: InputData):
    try:
        result = predict(data.features)
        return {"prediction": result}
    except Exception as e:
        return HTTPException(status_code=400, detail=str(e))
    
