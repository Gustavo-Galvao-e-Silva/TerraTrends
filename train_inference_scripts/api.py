from fastapi import FastAPI
from score_engine import predict
from pydantic import BaseModel

app = FastAPI()

class PredictionRequest(BaseModel):
    sector: str
    revenue: float
    employee_count: int
    founding_year: int
    horizon: str

@app.post("/analysis/predict")
def make_pred(request: PredictionRequest):
    econ_data = "data/merged_data_v2.csv"
    model_path = "lstm_model_v2.pt"
    return predict(request.sector, request.revenue, request.employee_count,
                   request.founding_year, econ_data, model_path, request.horizon)

