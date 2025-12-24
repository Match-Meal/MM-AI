# MatchMeal (MM) AI Project

## 📋 프로젝트 개요

FastAPI를 활용한 AI 마이크로서비스 프로젝트로, **MatchMeal** 서비스에 필요한 **이미지 기반 음식 분석** 및 **영양 코칭 챗봇** 기능을 제공합니다. RAG(Retrieval-Augmented Generation) 기술을 도입하여 신뢰성 높은 건강 정보를 답변합니다.

### 진행 일정

- **진행일자**: 2025.11 ~ 2025.12
- **개발 환경**: Python 3.12+, FastAPI

---

## 🎯 목표

- ✅ FastAPI를 활용한 고성능 비동기 API 서버 구축
- ✅ LangChain 및 OpenAI API를 활용한 LLM 애플리케이션 개발
- ✅ RAG(검색 증강 생성) 파이프라인 구축을 통한 할루시네이션 최소화
- ✅ Vector DB(ChromaDB)를 활용한 효율적인 지식 검색 구현
- ✅ **Qwen2.5 VL 3B 모델**을 활용한 고정확도 식단 이미지 분석 구현

---

## 🛠 준비사항

### 1) 개발언어 및 툴

- **Language**: Python 3.10+
- **Framework**: FastAPI
- **Server**: Uvicorn
- **IDE**: VS Code / PyCharm
- **Virtual Env**: venv / Conda

### 2) 필수 라이브러리 / 알고리즘

- **Web**: FastAPI, Pydantic, Requests
- **LLM/AI**: LangChain, OpenAI, HuggingFace Transformers
- **Vector DB**: ChromaDB
- **ML/DL**: PyTorch, TorchVision
- **Image Processing**: Pillow, NumPy
- **Utils**: Python-dotenv (환경변수 관리)

### 3) AI 모델 및 데이터

- **LLM**: GPT-5.2 (Heavy) / GPT-4.1-mini (Fast) - Dual Model Strategy
- **Embedding**: OpenAI Embeddings / HuggingFace Embeddings
- **Vision**: **Qwen2.5 VL 3B** (AI-HUB 식단 학습 데이터 Fine-tuning)
- **Knowledge Base**: 영양학 가이드라인, 식품 영양 데이터베이스

---

## 📝 작업 순서

### 1단계: 환경 설정 및 기본 서버 구축

- Python 가상환경 구성 및 `requirements.txt` 의존성 설치
- FastAPI 기본 앱 생성 및 Hello World 테스트

### 2단계: 이미지 분석 API 구현

- 음식 사진 업로드 엔드포인트 구현 (`multipart/form-data`)
- **Qwen2.5 VL 3B 모델(AI-HUB 데이터 학습)**을 활용하여 음식 종류 및 양 추론
- 분석 결과를 JSON 형태로 반환

### 3단계: RAG 기반 챗봇 구현

- 영양학 지식 데이터 수집 및 전처리 (Text Split)
- ChromaDB에 임베딩 벡터 저장 (Vector Store 구축)
- 사용자 질문 -> 검색(Retriever) -> LLM 답변 생성 파이프라인 구현

### 4단계: Spring Boot 연동

- 백엔드 서버와의 통신을 위한 API 규격 정의
- 비동기 처리 및 에러 핸들링 고도화

### 5단계: 성능 최적화 및 테스팅

- 프롬프트 엔지니어링을 통한 답변 품질 개선
- 응답 속도 개선을 위한 캐싱 전략 고려

---

## 📋 요구사항 명세

### ✅ 기본(필수) 기능

#### 1. 음식 이미지 분석

| 기능             | 설명                                                     | 우선순위 |
| ---------------- | -------------------------------------------------------- | -------- |
| 이미지 업로드    | 사용자가 촬영한 음식 사진을 전송 받는 기능               | 필수     |
| 음식 인식        | 사진 속 음식의 종류(예: 김치찌개, 현미밥) 식별           | 필수     |
| 영양 정보 추론   | 식별된 음식의 칼로리 및 탄단지 비율 추정                 | 필수     |

#### 2. AI 영양 코칭

| 기능             | 설명                                                     | 우선순위 |
| ---------------- | -------------------------------------------------------- | -------- |
| 질문 답변        | "오늘 점심 추천해줘" 등 건강 관련 질문에 대한 답변       | 필수     |
| 식단 평가        | 사용자의 하루 식단 기록을 바탕으로 영양 밸런스 점수화    | 필수     |

### 🚀 심화 기능

#### 1. RAG 고도화

| 기능             | 설명                                                     | 우선순위 |
| ---------------- | -------------------------------------------------------- | -------- |
| 맞춤형 컨텍스트  | 사용자의 개인 건강 정보(알러지, 질병 등)를 반영한 답변   | 심화     |
| 출처 표시        | 답변의 근거가 되는 문서 출처(예: 보건복지부 가이드) 명시 | 심화     |

---

## 🏗 시스템 아키텍처 (AI Service)

### RAG Pipeline Flow

```
User Query
  ⬇️
[FastAPI Server]
  ⬇️
[Embedding Model]  <-- 질문을 벡터로 변환
  ⬇️
[ChromaDB]         <-- 유사한 문서 검색 (Retrieve)
  ⬇️
[LLM (OpenAI)]     <-- 질문 + 검색된 문서(Context) + 프롬프트
  ⬇️
Response Generation
```

---

## 📂 프로젝트 구조

```
MM-AI/
├── app/
│   ├── api/             # API 엔드포인트 (Routers)
│   ├── core/            # 설정 (Config, Security)
│   ├── services/        # 비즈니스 로직 (ChatService, VisionService)
│   ├── models/          # Pydantic 모델 (Schemas)
│   ├── utils/           # 유틸리티 (Prompt Templates 등)
│   └── main.py          # 앱 진입점
├── chroma_db/           # 벡터 DB 저장소
├── requirements.txt     # 의존성 목록
└── .env                 # 환경변수 (API Key 등)
```

---

## 📊 API 명세 (Swagger)

서버 실행 후 자동 생성된 문서를 확인할 수 있습니다.

- **Local**: `http://localhost:8000/docs`

### 주요 Endpoints

- `/ai/period-feedback`: 기간별 식단 정밀 분석 (Heavy Model)
- `/ai/recommend`: 맞춤 메뉴 추천 (Heavy Model)
- `/ai/meal-plan`: 기간별 식단표 생성 (Heavy Model)
- `/ai/chat`: 일반 대화 및 리포트 조회 (Fast Model)
- `/ai/history/{user_id}`: 대화 히스토리 조회

---

## 📤 결과 제출

### 산출물

1. **AI Model/Code**: FastAPI 프로젝트 소스코드
2. **Prompt Templates**: 시스템 프롬프트 및 Few-shot 예시
3. **Requirement Docs**: 라이브러리 버전 명시 문서

---

## 📚 참고 자료

### 문서

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [LangChain Python Docs](https://python.langchain.com/)
- [OpenAI API Reference](https://platform.openai.com/docs/api-reference)

### 개발 팁

- **Uvicorn**: 개발 시에는 `--reload` 옵션을 사용하여 코드 변경 사항 즉시 반영
- **Typing**: Python Type Hint를 적극 활용하여 코드 안정성 확보

---

## 검토 사항

- [ ] Python 3.10 이상 설치 확인
- [ ] `.env` 파일에 `OPENAI_API_KEY` 설정 확인
- [ ] `pip install -r requirements.txt` 에러 없이 완료 확인
- [ ] API 서버 실행 및 Swagger UI 접속 확인
