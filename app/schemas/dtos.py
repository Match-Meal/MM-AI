# 데이터의 형태(DTO)를 정의

from pydantic import BaseModel
from typing import List, Optional

# AI 분석 결과 응답 형식
class FoodAnalysisResponse(BaseModel):
    candidates: List[str]  # ["김치찌개", "부대찌개", "김치찜"]
    best_candidate: str    # "김치찌개" (1순위)
    
# (나중에 챗봇용 DTO도 여기에 추가하면 됩니다)
class ChatRequest(BaseModel):
    user_id: int
    message: str