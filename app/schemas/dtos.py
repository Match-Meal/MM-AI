# 데이터의 형태(DTO)를 정의

from pydantic import BaseModel
from typing import List, Optional

# AI 분석 결과 응답 형식
class FoodAnalysisResponse(BaseModel):
    candidates: List[str]  # ["김치찌개", "부대찌개", "김치찜"]
    best_candidate: str    # "김치찌개" (1순위)

class UserProfile(BaseModel):
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
    user_profile: UserProfile
    current_intake: IntakeSummary
    meal_type: str
    flavors: List[str] = [] # 매운맛, 짠맛 등



# [API 3 요청] 일반 대화 (히스토리 포함)
class ChatRequest(BaseModel):
    user_profile: UserProfile
    history: List[dict] # [{"role": "user", "content": "..."}, ...]
    message: str

# 2. 기간 분석용 모델
class PeriodInfo(BaseModel):
    start_date: str
    end_date: str
    total_days: int
    recorded_meals: int

class PeriodNutritionStats(BaseModel):
    avg_calories: float
    total_sodium: float
    total_sugar: float

# [API 1 요청] 기간별 식단 피드백
class PeriodFeedbackRequest(BaseModel):
    user_profile: UserProfile
    period_info: PeriodInfo
    nutrition_stats: PeriodNutritionStats
    menu_list: List[str]

# [API New] 기간별 식단 추천
class MealPlanRequest(BaseModel):
    user_profile: UserProfile
    period_info: PeriodInfo
    flavors: List[str] = []

