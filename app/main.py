from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.router import api_router

app = FastAPI(title="TalentAgent API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://[::1]:5173",
        "http://[::1]:5174",
        "https://chhavi-6agents-graph.vercel.app",
    ],
    allow_origin_regex=r"^https:\/\/.*\.vercel\.app$|^https?:\/\/(localhost|127\.0\.0\.1|\[::1\])(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router, prefix="/api/v1")

@app.get("/")
async def root():
    return {"status": "ok", "service": "talent-agent", "message": "TalentGraph API is live and healthy"}

@app.get("/health")
async def health():
    return {"status": "ok", "service": "talent-agent"}
