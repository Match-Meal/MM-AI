import asyncio
from fastapi import FastAPI, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from contextlib import asynccontextmanager
from app.core.ai_model import load_model
from app.routers import vision
from app.schemas.dtos import PeriodFeedbackRequest, RecommendRequest, ChatRequest, MealPlanRequest
from app.services.agent import coach
from app.services.vector_store import tool_store
from fastapi.middleware.cors import CORSMiddleware
from app.services.history_service import history_service
from app.core.database import AsyncSessionLocal
from datetime import datetime, date, timedelta

# --- Helper: Stream & Save ---
async def stream_and_save(generator, user_id: int, ai_type: str, question: str, ref_date=None):
    """
    제너레이터의 출력을 스트리밍하면서, 완료 후 DB에 저장합니다.
    """
    full_answer = ""
    try:
        async for chunk in generator:
            full_answer += chunk
            yield chunk
    except Exception as e:
        print(f"Streaming Error: {e}")
        full_answer += f"\n[Error] {e}"
        yield f"\n[Error] {e}"
    finally:
        # 스트리밍 완료 후 DB 저장 (비동기 세션 별도 생성)
        if full_answer and user_id:
            print(f"💾 Saving History... User={user_id}, Type={ai_type}")
            async with AsyncSessionLocal() as session:
                await history_service.save_chat_history(
                    session, user_id, ai_type, question, full_answer, ref_date
                )

# 1. 수명 주기(Lifespan) 관리: 서버 켜질 때 모델 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MatchMeal AI 서버 시작 중...")
    
    async def initialize_data():
        try:
            print("🔍 AI 모델 로딩 시도...")
            load_model()
            print("✅ AI 모델 로딩 완료 (RAG 모드)")
            
            print("💾 벡터 데이터베이스 데이터 로딩 및 인덱싱 시작 (백그라운드)...")
            from app.services.vector_store import food_store
            food_store.load_from_csvs()
            print("✅ 벡터 데이터베이스 준비 완료")
            
            print("🛠️ 도구 인덱싱 중...")
            tool_store.index_tools(coach.all_tools)
            print("✅ 도구 인덱싱 완료")
            print("✨ 모든 초기화 작업이 백그라운드에서 완료되었습니다.")
        except Exception as e:
            print(f"❌ 백그라운드 초기화 중 오류 발생: {e}")

    # 초기화 작업을 백그라운드 태스크로 시작
    init_task = asyncio.create_task(initialize_data())
    
    try:
        print("� API 서비스 시작 준비 완료 (초기화는 백그라운드에서 진행 중)")
        yield
    except BaseException as b:
        if not isinstance(b, asyncio.CancelledError):
            print(f"⚠️ lifespan 종료 중 예외 발생: {type(b).__name__}: {b}")
    finally:
        if not init_task.done():
            init_task.cancel()
        print("👋 AI 서버가 종료됩니다.")

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

# [API 1] 기간별 식단 피드백 -> Heavy Model
@app.post("/ai/period-feedback")
async def period_feedback(req: PeriodFeedbackRequest):
    # Handle missing user profile
    user_data = req.user_profile.model_dump() if req.user_profile else {}
    user_id = req.user_profile.user_id if req.user_profile else 0

    # Handle missing nutrition_stats
    avg_cal = req.nutrition_stats.avg_calories if req.nutrition_stats else 0.0
    total_sod = req.nutrition_stats.total_sodium if req.nutrition_stats else 0.0
    total_sug = req.nutrition_stats.total_sugar if req.nutrition_stats else 0.0
    
    context = f"""
    [요청: 기간별 식단 정밀 분석]
    기간: {req.period_info.start_date} ~ {req.period_info.end_date} (총 {req.period_info.total_days}일)
    기록된 끼니 수: {req.period_info.recorded_meals}끼
    
    [영양 통계]
    - 일 평균 칼로리: {avg_cal:.1f}kcal
    - 기간 총 나트륨: {total_sod:.1f}mg
    - 기간 총 당류: {total_sug:.1f}g
 
    [섭취한 메뉴 목록]
    {', '.join(req.menu_list) if req.menu_list else '기록된 메뉴 없음'}
 
    위 데이터를 바탕으로 사용자의 식습관을 평가하고 개선점을 알려주세요.
    """
    
    # use_fast_model=False (Heavy)
    generator = coach.stream_agent_response(context, user_data, use_fast_model=False)
    
    # Question Text for DB
    q_text = f"기간분석 요청 ({req.period_info.start_date}~{req.period_info.end_date})"

    return StreamingResponse(
        stream_and_save(generator, user_id, "FEEDBACK", q_text, date.today()),
        media_type="text/plain"
    )

# [API 2] 메뉴 추천 -> Heavy Model
@app.post("/ai/recommend")
async def recommend(req: RecommendRequest):
    # Handle defaults
    user_data = req.user_profile.model_dump() if req.user_profile else {}
    user_id = req.user_profile.user_id if req.user_profile else 0
    cur_cal = req.current_intake.calories if req.current_intake else 0
    cur_sod = req.current_intake.sodium if req.current_intake else 0
    cur_sug = req.current_intake.sugar if req.current_intake else 0
    
    context = f"""
    [요청: 맞춤 메뉴 추천]
    사용자가 선택한 끼니: {req.meal_type}
    
    [오늘 현재까지 섭취량]
    - 칼로리: {cur_cal}kcal
    - 나트륨: {cur_sod}mg
    - 당류: {cur_sug}g
    
    [사용자 취향]
    {', '.join(req.flavors) if req.flavors else "특별한 취향 없음"}
    
    사용자의 프로필(질병, 알레르기)과 오늘 섭취량을 고려하여,
    부족한 영양소는 채우고 과잉된 영양소는 피할 수 있는 메뉴를 추천해주세요.
    """
    
    # use_fast_model=False (Heavy)
    generator = coach.stream_agent_response(context, user_data, flavors=req.flavors, use_fast_model=False)
    
    q_text = f"메뉴 추천 ({req.meal_type}, {', '.join(req.flavors)})"

    return StreamingResponse(
        stream_and_save(generator, user_id, "RECOMMENDATION", q_text, date.today()),
        media_type="text/plain"
    )

# [API New] 기간별 식단 추천 -> Heavy Model
@app.post("/ai/meal-plan")
async def meal_plan(req: MealPlanRequest):
    # Handle user profile
    user_data = req.user_profile.model_dump() if req.user_profile else {}
    user_id = req.user_profile.user_id if req.user_profile else 0

    context = f"""
    [요청: 맞춤 식단표 생성]
    기간: {req.period_info.start_date} ~ {req.period_info.end_date} (총 {req.period_info.total_days}일)
    
    [사용자 취향]
    {', '.join(req.flavors) if req.flavors else "특별한 취향 없음"}
    
    위 기간 동안 사용자가 실천할 수 있는 구체적인 식단표를 짜주세요.
    - **도구 사용 필수:** `recommend_food_from_db` 도구를 사용하여 각 끼니에 적합한 메뉴를 찾아주세요.
    - **구성:** 아침, 점심, 저녁 메뉴와 칼로리를 포함해야 합니다.
    - **형식:** 날짜별로 구분하여 보기 좋게 출력해주세요. (반드시 리스트 형식을 사용하세요)
    """
    
    # use_fast_model=False (Heavy)
    generator = coach.stream_agent_response(context, user_data, flavors=req.flavors, use_fast_model=False)
    
    q_text = f"식단표 생성 ({req.period_info.start_date}~{req.period_info.end_date})"

    return StreamingResponse(
        stream_and_save(generator, user_id, "MEAL_PLAN", q_text, date.today()),
        media_type="text/plain"
    )

# [API 3] 일반 대화 (히스토리 포함) -> Fast Model
@app.post("/ai/chat")
async def chat(req: ChatRequest):
    # Handle user profile
    user_data = req.user_profile.model_dump() if req.user_profile else {}
    user_id = req.user_profile.user_id if req.user_profile else 0

    # use_fast_model=True (Fast)
    generator = coach.stream_agent_response(req.message, user_data, history=req.history, use_fast_model=True, persona=req.persona)
    
    return StreamingResponse(
        stream_and_save(generator, user_id, "CHAT", req.message, date.today()),
        media_type="text/plain"
    )

@app.get("/ai/history/{user_id}")
async def get_history(user_id: int):
    """
    사용자의 AI 대화 히스토리를 DB에서 조회하여 반환합니다.
    """
    async with AsyncSessionLocal() as session:
        return {"data": await history_service.get_chat_history(session, user_id)}