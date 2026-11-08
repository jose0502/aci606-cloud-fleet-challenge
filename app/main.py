from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import func
import time
import joblib
import pandas as pd
from pathlib import Path

from .db import SessionLocal, engine, Base
from .models import Telemetry

Base.metadata.create_all(bind=engine)

# 1. CREAMOS LA APP
app = FastAPI(title="ACI606 - Cloud Fleet Challenge")

# --- CARGAR EL MODELO ML ---
MODEL_PATH = Path("models/model.joblib")
if not MODEL_PATH.exists():
    raise RuntimeError("Modelo no encontrado. Ejecuta python ml/train_model.py")

model = joblib.load(MODEL_PATH)
MODEL_VERSION = "v1"

FEATURES = [
    "speed", "battery", "motor_temp", 
    "vibration", "motor_current", "obstacle_distance"
]

# --- MÉTRICAS (Observabilidad) ---
METRICS = {
    "telemetry_requests": 0,
    "prediction_requests": 0,
    "errors": 0,
    "total_inference_ms": 0.0,
}
START_TIME = time.time()

@app.middleware("http")
async def count_requests(request: Request, call_next):
    response = await call_next(request)
    if response.status_code >= 400:
        METRICS["errors"] += 1
    return response

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- MODELOS PYDANTIC ---
class TelemetryIn(BaseModel):
    robot_id: str
    timestamp: float
    x: float
    y: float
    speed: float
    battery: float
    motor_temp: float
    vibration: float
    motor_current: float
    obstacle_distance: float

class PredictIn(BaseModel):
    robot_id: str
    timestamp: float
    x: float
    y: float
    speed: float
    battery: float
    motor_temp: float
    vibration: float
    motor_current: float
    obstacle_distance: float

# --- ENDPOINTS ---

@app.get("/health")
def health(db: Session = Depends(get_db)):
    # Verificamos la conexión a la base de datos contando registros
    total_registros = db.query(Telemetry).count()
    return {
        "status": "ok", 
        "model_version": MODEL_VERSION,
        "storage": "Neon PostgreSQL",
        "total_registros": total_registros
    }

@app.post("/telemetry")
def post_telemetry(payload: TelemetryIn, db: Session = Depends(get_db)):
    METRICS["telemetry_requests"] += 1
    new_record = Telemetry(**payload.model_dump())
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return {"status": "ok", "id": new_record.id}

@app.get("/robots")
def get_robots(db: Session = Depends(get_db)):
    subquery = db.query(
        Telemetry.robot_id, 
        func.max(Telemetry.timestamp).label("max_ts")
    ).group_by(Telemetry.robot_id).subquery()

    results = db.query(Telemetry).join(
        subquery,
        (Telemetry.robot_id == subquery.c.robot_id) & 
        (Telemetry.timestamp == subquery.c.max_ts)
    ).order_by(Telemetry.robot_id.asc()).all() # <--- Agregamos el orden aquí
    
    return results

@app.get("/robots/{robot_id}/history")
def get_history(robot_id: str, db: Session = Depends(get_db)):
    results = db.query(Telemetry).filter(
        Telemetry.robot_id == robot_id
    ).order_by(Telemetry.timestamp.asc()).all()
    if not results:
        raise HTTPException(status_code=404, detail="Robot no encontrado")
    return results

@app.get("/robots/{robot_id}/stats")
def get_robot_stats(robot_id: str, db: Session = Depends(get_db)):
    # La base de datos hace los cálculos matemáticos (AVG, MIN, MAX) directamente
    stats = db.query(
        func.count(Telemetry.id).label("muestras"),
        func.avg(Telemetry.battery).label("bateria_promedio"),
        func.max(Telemetry.motor_temp).label("temp_maxima"),
        func.avg(Telemetry.vibration).label("vibracion_promedio")
    ).filter(Telemetry.robot_id == robot_id).first()

    if not stats or stats.muestras == 0:
        raise HTTPException(status_code=404, detail="Robot sin datos")

    return {
        "robot_id": robot_id,
        "muestras_totales": stats.muestras,
        "bateria_promedio": round(stats.bateria_promedio, 2),
        "temperatura_maxima": round(stats.temp_maxima, 2),
        "vibracion_promedio": round(stats.vibracion_promedio, 2)
    }

@app.post("/predict")
def predict(payload: PredictIn):
    METRICS["prediction_requests"] += 1
    start_time = time.perf_counter()
    
    input_data = {k: getattr(payload, k) for k in FEATURES}
    X = pd.DataFrame([input_data])
    
    prediction = model.predict(X)[0]
    probabilities = model.predict_proba(X)[0]
    confidence = float(max(probabilities))
    
    end_time = time.perf_counter()
    inference_ms = round((end_time - start_time) * 1000, 2)
    
    METRICS["total_inference_ms"] += inference_ms
    
    return {
        "robot_id": payload.robot_id,
        "predicted_status": str(prediction),
        "confidence": round(confidence, 4),
        "model_version": MODEL_VERSION,
        "inference_ms": inference_ms
    }

@app.get("/metrics")
def get_metrics():
    uptime_seconds = time.time() - START_TIME
    avg_latency = 0.0
    if METRICS["prediction_requests"] > 0:
        avg_latency = METRICS["total_inference_ms"] / METRICS["prediction_requests"]
        
    return {
        "telemetry_requests": METRICS["telemetry_requests"],
        "prediction_requests": METRICS["prediction_requests"],
        "errors": METRICS["errors"],
        "avg_prediction_latency_ms": round(avg_latency, 2),
        "uptime_seconds": round(uptime_seconds, 2)
    }

# --- DASHBOARD ---
@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    dashboard_path = Path("dashboard/index.html")
    if dashboard_path.exists():
        return dashboard_path.read_text(encoding="utf-8")
    return "<h1>Dashboard no encontrado</h1>"