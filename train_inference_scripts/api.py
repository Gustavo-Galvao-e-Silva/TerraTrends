from fastapi import FastAPI
from score_engine import predict
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class PredictionRequest(BaseModel):
    sector: str
    revenue: float
    employee_count: int
    founding_year: int
    horizon: str

@app.post("/analysis/predict")
def make_pred(request: PredictionRequest):
    econ_data = "data/merged_data_v2.csv"
    static_forecasts_path = "train_inference_scripts/static_forecasts.csv"
    return predict(request.sector, request.revenue, request.employee_count,
                   request.founding_year, econ_data, static_forecasts_path, request.horizon)

