from fastapi import FastAPI
from contextlib import asynccontextmanager
from app.core.ai_model import load_model
from app.routers import vision

# 1. 수명 주기(Lifespan) 관리: 서버 켜질 때 모델 로드
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("🚀 MatchMeal AI Server Starting...")
    load_model()  # 여기서 모델 로딩 (시간 좀 걸림)
    yield
    print("👋 Server Shutting Down...")

# 2. 앱 생성
app = FastAPI(
    title="MatchMeal AI Server",
    description="Qwen2.5-VL based Food Analysis API",
    version="1.0.0",
    lifespan=lifespan
)

# 3. 라우터 등록
app.include_router(vision.router)

@app.get("/")
def health_check():
    return {"status": "ok", "server": "MatchMeal AI is ready"}