from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.ai_model import load_model
from app.routers import vision
from app.schemas.dtos import PeriodFeedbackRequest, RecommendRequest, ChatRequest, MealPlanRequest
from app.services.agent import coach
from fastapi.middleware.cors import CORSMiddleware


# 1. 수명 주기(Lifespan) 관리: 서버 켜질 때 모델 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MatchMeal AI Server Starting...")
    # [RAG 테스트 모드] 이미지 추론 모델 로딩 생략
    # load_model()
    print("⚠️ 이미지 모델(Qwen) 로딩이 비활성화되었습니다. (RAG 기능만 모드)")
    
    # 벡터 DB 초기화 및 데이터 적재
    from app.services.vector_store import food_store
    food_store.load_from_csvs()
    
    yield
    print("👋 Server Shutting Down...")

# 2. 앱 생성
app = FastAPI(
    title="MatchMeal AI Server",
    description="Qwen2.5-VL based Food Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. 라우터 등록
app.include_router(vision.router)


@app.get("/")
def health_check():
    return {"status": "ok", "msg": "MatchMeal AI Ready"}

# [API 1] 기간별 식단 피드백
@app.post("/ai/period-feedback")
async def period_feedback(req: PeriodFeedbackRequest):
    context = f"""
    [요청: 기간별 식단 정밀 분석]
    기간: {req.period_info.start_date} ~ {req.period_info.end_date} (총 {req.period_info.total_days}일)
    기록된 끼니 수: {req.period_info.recorded_meals}끼
    
    [영양 통계]
    - 일 평균 칼로리: {req.nutrition_stats.avg_calories:.1f}kcal
    - 기간 총 나트륨: {req.nutrition_stats.total_sodium:.1f}mg
    - 기간 총 당류: {req.nutrition_stats.total_sugar:.1f}g

    [섭취한 메뉴 목록]
    {', '.join(req.menu_list)}

    위 데이터를 바탕으로 사용자의 식습관을 평가하고 개선점을 알려주세요.
    """
    
    # 일반 분석은 history/flavors 없이 호출
    return {"result": coach.run_agent(context, req.user_profile.model_dump())}

# [API 2] 메뉴 추천
@app.post("/ai/recommend")
async def recommend(req: RecommendRequest):
    context = f"""
    [요청: 맞춤 메뉴 추천]
    사용자가 선택한 끼니: {req.meal_type}
    
    [오늘 현재까지 섭취량]
    - 칼로리: {req.current_intake.calories}kcal
    - 나트륨: {req.current_intake.sodium}mg
    - 당류: {req.current_intake.sugar}g
    
    [사용자 취향]
    {', '.join(req.flavors) if req.flavors else "특별한 취향 없음"}
    
    사용자의 프로필(질병, 알레르기)과 오늘 섭취량을 고려하여,
    부족한 영양소는 채우고 과잉된 영양소는 피할 수 있는 메뉴를 추천해주세요.
    """
    
    return {"result": coach.run_agent(context, req.user_profile.model_dump(), flavors=req.flavors)}

# [API New] 기간별 식단 추천
@app.post("/ai/meal-plan")
async def meal_plan(req: MealPlanRequest):
    context = f"""
    [요청: 맞춤 식단표 생성]
    기간: {req.period_info.start_date} ~ {req.period_info.end_date} (총 {req.period_info.total_days}일)
    
    [사용자 취향]
    {', '.join(req.flavors) if req.flavors else "특별한 취향 없음"}
    
    위 기간 동안 사용자가 실천할 수 있는 구체적인 식단표를 짜주세요.
    - **도구 사용 필수:** `recommend_food_from_db` 도구를 사용하여 각 끼니에 적합한 메뉴를 찾아주세요.
    - **구성:** 아침, 점심, 저녁 메뉴와 칼로리를 포함해야 합니다.
    - **형식:** 날짜별로 구분하여 보기 좋게 출력해주세요. (마크다운 표 또는 리스트 형식)
    """
    
    return {"result": coach.run_agent(context, req.user_profile.model_dump(), flavors=req.flavors)}

# [API 3] 일반 대화 (히스토리 포함)
@app.post("/ai/chat")
async def chat(req: ChatRequest):
    # 히스토리 기반 대화
    return {"result": coach.run_agent(req.message, req.user_profile.model_dump(), history=req.history)}