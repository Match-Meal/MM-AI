# 데이터의 형태(DTO)를 정의

from pydantic import BaseModel
from typing import List, Optional

# AI 분석 결과 응답 형식
class FoodAnalysisResponse(BaseModel):
    candidates: List[str]  # ["김치찌개", "부대찌개", "김치찜"]
    best_candidate: str    # "김치찌개" (1순위)

class UserProfile(BaseModel):
    user_id: Optional[int] = None
    name: str
    age: int
    gender: str
    height_cm: float = 0.0
    weight_kg: float = 0.0
    bmi: float
    bmi_status: str
    allergies: Optional[str] = ""
    diseases: Optional[str] = ""

class IntakeSummary(BaseModel):
    calories: float
    sodium: float
    sugar: float

# [API 2 요청] 메뉴 추천
class RecommendRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    current_intake: Optional[IntakeSummary] = None
    meal_type: str
    flavors: List[str] = [] # 매운맛, 짠맛 등

# [API 3 요청] 일반 대화 (히스토리 포함)
class ChatRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    history: List[dict] = [] # [{"role": "user", "content": "..."}, ...]
    message: str
    persona: str = "coach"  # 'coach' | 'friend'

# 2. 기간 분석용 모델
class PeriodInfo(BaseModel):
    start_date: str
    end_date: str
    total_days: int = 0
    recorded_meals: int = 0

class PeriodNutritionStats(BaseModel):
    avg_calories: float = 0.0
    total_sodium: float = 0.0
    total_sugar: float = 0.0

# [API 1 요청] 기간별 식단 피드백
class PeriodFeedbackRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    period_info: PeriodInfo
    nutrition_stats: Optional[PeriodNutritionStats] = None
    menu_list: List[str] = []

# [API New] 기간별 식단 추천
class MealPlanRequest(BaseModel):
    user_profile: Optional[UserProfile] = None
    period_info: PeriodInfo
    flavors: List[str] = []

