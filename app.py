from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from joblib import load
import pandas as pd
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware
from _binance import generate_realtime_candlestick_data

app = FastAPI()

# Allow all origins (replace "*" with the specific origins you want to allow)
origins = [
    "https://ml-react.onrender.com"
   ]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

symbol = 'ETHUSDT'  # Replace with your desired trading pair
interval = '1m'    # Replace with your desired time interval
contract = 'PERPETUAL'
limit = 200

# output signals 
# Sell - 2
# Buy - 0
# Hold - 1

# Load the trained machine learning model
model = load('trained_model.joblib')

class PredictionInput(BaseModel):
    macd_crossover: Optional[float]
    last_candle: Optional[float]
    shooting_star: Optional[float]
    ma_trend: Optional[float]
    macd_turning_point: Optional[float]
    rsi: Optional[float]
    buy_sell_pressure: Optional[float]

@app.post("/predict")
async def predict(data : Optional[PredictionInput] = None):

    try:
        if data:
            # Convert input data to DataFrame
            input_data = pd.DataFrame([data.dict()])

            # Make prediction
            prediction = model.predict(input_data)

            # Return the prediction as JSON
            return JSONResponse(content={"prediction": int(prediction[0])})

        else:
            # get the lts data from the binance API and make the prediction
            signals = await generate_realtime_candlestick_data(symbol, interval, contract, limit)

            input_data = pd.DataFrame([signals])

            # Make prediction
            prediction = model.predict(input_data)

            # Return the prediction as JSON
            return JSONResponse(content={"prediction": int(prediction[0])})
    except HTTPException as e:
            return JSONResponse(content={"error": e.detail}, status_code=e.status_code)