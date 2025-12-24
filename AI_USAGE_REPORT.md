# 🤖 AI Server Usage & Architecture Report

## 1. 개요 (Overview)
본 문서는 MM-AI 서버에서 사용되는 **이미지 분석(Vision AI)** 및 **챗봇 코칭(Chatbot AI)** 서비스의 기술적인 구현 내용과 AI 모델 활용 현황을 정리한 보고서입니다.

---

## 2. 이미지 분석 (Vision AI Service)
음식 사진을 업로드하면 자동으로 음식의 이름을 분석하고 후보군을 제시합니다.

### 2.1 사용 모델 (Model Architecture)
- **Base Model (기반 모델)**: `Qwen/Qwen2.5-VL-3B-Instruct`
    - 다목적 비전-언어 모델(VLM)로, 이미지와 텍스트를 동시에 이해합니다.
- **Fine-tuning (미세 조정)**: `PeftModel` (LoRA Adapter)
    - **경로**: `./models/food_adapter_v1.0`
    - 한국 음식 데이터에 특화된 커스텀 어댑터(LoRA)가 적용되어 있어 한식 인식률이 높습니다.
- **Hardware Acceleration**:
    - `CUDA` (NVIDIA GPU), `MPS` (Apple Silicon), `CPU`를 자동 감지하여 최적화된 장치에서 추론합니다.

### 2.2 처리 프로세스 (Processing Pipeline)
1. **이미지 전처리**:
   - `MAX_IMAGE_DIMENSION = 1024` (메모리 최적화를 위해 리사이징)
   - RGB 포맷 변환
2. **프롬프트 엔지니어링**:
   - 페르소나: "한국 음식 전문가"
   - JSON 포맷 강제화: `best_candidate`와 `candidates` 필드를 포함한 JSON 출력을 유도합니다.
3. **추론 (Inference)**:
   - `process_vision_info`를 통해 이미지 텐서 변환
   - `max_new_tokens=256` 제한으로 빠른 응답 유도
4. **후처리 (Post-processing)**:
   - Markdown 코드 블록(` ```json `) 제거 및 파싱
   - JSON 파싱 실패 시, AI의 단답형 텍스트 응답을 `best_candidate`로 사용하는 Fallback 로직 적용

### 2.3 주요 코드
- **서비스 로직**: `app/services/vision_service.py`
- **모델 로더**: `app/core/ai_model.py`

---

## 3. 챗봇 코칭 (Food Coaching Agent)
사용자의 건강 정보와 식사 기록을 바탕으로 개인화된 영양 가이드를 제공하는 에이전트입니다.

### 3.1 아키텍처 (Architecture)
- **프레임워크**: `LangChain` (Tool Calling Agent)
- **Dual LLM Strategy**: 성능과 비용 최적화를 위해 역할에 따라 모델을 분리했습니다.
    - **Fast LLM (`gpt-4.1-mini`)**:
        - 역할: 도구(Tool) 선택, 일반 대화, 단순 질의응답.
        - 특징: 응답 속도가 빠르고 비용이 저렴합니다.
    - **Heavy LLM (`gpt-5.2`)**:
        - 역할: 심층 분석, 복잡한 식단 계획, 정밀 추론.
        - 특징: 높은 지능을 가지며 스트리밍(Streaming)으로 체감 대기 시간을 줄였습니다.

### 3.2 에이전트 기능 (Agent Capabilities)
챗봇은 **14가지 전문 도구(Tools)** 중 필요한 것을 스스로 선택하여 답변합니다.

#### 🛠️ 주요 도구 (Core Tools)
1. **건강/영양 분석**: BMI, BMR, 일일 권장량 대비 섭취량 분석.
2. **음식 DB 검색/추천**: 고혈압, 당뇨 등 질환별 필터링 검색 (RAG).
3. **운동 칼로리 계산**: METs 기반 소모 칼로리 계산.
4. **음식 간 비교**: 두 음식의 영양 성분 1:1 비교.
5. **장보기 리스트**: 식단 텍스트를 장보기 목록으로 변환.

#### 🌟 확장 도구 (Expanded Tools)
6. **제철 음식 추천**: 월별 제철 식재료 추천.
7. **증상 완화 음식**: 감기, 소화불량 등 증상별 푸드 테라피.
8. **레시피 절차**: 주요 요리의 단계별 조리법 안내.
9. **음식 궁합**: 함께 먹으면 좋은/나쁜 음식 정보.
10. **유지 칼로리 계산**: 활동량(Activity Level)을 고려한 유지 열량 계산.
11. **건강 대체식**: 라면, 치킨 등 고칼로리 음식의 건강한 대체재 제안.
12. **수분 섭취 가이드**: 체중 기반 물 권장량 계산.
13. **상황별 간식**: 공부, 운동 전후 등 상황에 맞는 간식 추천.
14. **영양 결핍 분석**: 눈떨림, 입병 등 증상으로 부족 영양소 추론.

### 3.3 개인화 및 페르소나 (Personalization)
- **시스템 프롬프트**: "냠냠코치"라는 이름으로 활동하며, 사용자의 프로필(나이, 질환, 알레르기)을 항상 컨텍스트에 포함합니다.
- **페르소나 모드**:
    - `coach`: 존댓말, 전문적, 공손함.
    - `friend`: 반말, 친근함, 유머러스함.
- **응답 형식**: 사용자가 읽기 편하도록 항상 **"3줄 요약"**을 먼저 제시하고 상세 내용을 설명합니다.

### 3.4 기술적 특징
1. **Hybrid Tool Selection**:
   - 도구 검색 시 Vector Search(Recall)와 LLM(Precision)을 결합하여 정확도를 높였습니다.
   - 도구 개수가 적을 때는 LLM에게 전체 목록을 주어 선택하게 함으로써 ChromaDB의 인덱싱 오류를 회피합니다.
2. **Startup Optimization**:
   - `lifespan` 이벤트를 활용하여 서버 시작 시 AI 모델 로드, DB 인덱싱을 백그라운드(`asyncio.create_task`)에서 수행합니다.
   - 이를 통해 서버 부팅 속도를 획기적으로 단축했습니다.

---

## 4. API 서비스 흐름 (Service Flow)

| 엔드포인트 | 목적 | 사용 모델 | 특징 |
| :--- | :--- | :--- | :--- |
| **`/ai/period-feedback`** | 기간별 식단 정밀 분석 | **Heavy** (`gpt-5.2`) | 사용자의 장기간 식습관을 분석하고 개선점을 제안합니다. |
| **`/ai/recommend`** | 맞춤 메뉴 추천 | **Heavy** (`gpt-5.2`) | 현재 섭취량과 영양 밸런스를 고려하여 최적의 메뉴를 추천합니다. |
| **`/ai/meal-plan`** | 식단표 생성 | **Heavy** (`gpt-5.2`) | 사용자의 취향과 목표에 맞는 구체적인 N일치 식단표를 생성합니다. |
| **`/ai/chat`** | 일반 대화 (상담) | **Fast** (`gpt-4.1-mini`) | 대화 맥락(`history`)을 기억하며 가볍고 빠른 상담을 제공합니다. |
