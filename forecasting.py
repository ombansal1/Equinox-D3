# forecasting.py
import pandas as pd
from prophet import Prophet
from datetime import datetime, timedelta

def forecast_mood(daily_mood):
    """
    Takes daily_mood = [{"date": "YYYY-MM-DD", "avg_compound": float}, ...]
    Returns next 7 days of mood forecast and badge/support message.
    """
    if len(daily_mood) < 3:
        return {
            "forecast": [],
            "badge": None,
            "message": "Not enough data to forecast yet."
        }

    # Convert to DataFrame for Prophet
    df = pd.DataFrame(daily_mood)
    df.rename(columns={"date": "ds", "avg_compound": "y"}, inplace=True)
    df["ds"] = pd.to_datetime(df["ds"])

    # Initialize and fit model
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
        changepoint_prior_scale=0.05
    )
    model.fit(df)

    # Predict next 7 days
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)

    # Extract relevant values
    pred = forecast[["ds", "yhat"]].tail(7)
    forecast_values = [
        {"date": d.strftime("%Y-%m-%d"), "predicted_mood": round(float(y), 3)}
        for d, y in zip(pred["ds"], pred["yhat"])
    ]

    # Compare last actual vs predicted
    recent_actual = df["y"].tail(7).mean()
    recent_forecast = pred["predicted_mood"].mean() if "predicted_mood" in pred else pred["yhat"].mean()

    # Badge / feedback logic
    diff = recent_actual - recent_forecast
    if diff >= 0.2:
        badge = "🥇 Beat the Forecast"
        message = "You're doing great — keep it up!"
    elif diff >= -0.1:
        badge = "🌈 Staying Balanced"
        message = "You're maintaining your emotional balance."
    else:
        badge = None
        message = "This week seems tough — we're here to support you 💛"

    return {
        "forecast": forecast_values,
        "badge": badge,
        "message": message
    }
