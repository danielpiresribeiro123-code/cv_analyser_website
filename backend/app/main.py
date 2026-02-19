from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes import pipeline

app = FastAPI(title="Gyro - CV Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],)

app.include_router(pipeline.router)
