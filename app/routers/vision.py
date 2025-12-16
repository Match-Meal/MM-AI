# 이미지 분석 API

from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.vision_service import analyze_food_image
from app.schemas.dtos import FoodAnalysisResponse

router = APIRouter(prefix="/vision", tags=["Vision AI"])

@router.post("/analyze", response_model=FoodAnalysisResponse)
async def analyze_food(file: UploadFile = File(...)):
    """
    이미지를 업로드하면 음식 이름을 분석하여 반환합니다.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일만 업로드 가능합니다.")
        
    try:
        result = await analyze_food_image(file)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))