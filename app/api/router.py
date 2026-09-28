from fastapi import APIRouter
from app.api.mandates import router as mandates_router
from app.api.candidates import router as candidates_router
from app.api.analysis import router as analysis_router
from app.api.github_mcp import router as github_mcp_router

api_router = APIRouter()
api_router.include_router(mandates_router, prefix="/mandates", tags=["mandates"])
api_router.include_router(candidates_router, prefix="/candidates", tags=["candidates"])
api_router.include_router(analysis_router, prefix="/analysis-runs", tags=["analysis"])
api_router.include_router(github_mcp_router, prefix="/github", tags=["github"])
