# MM-AI

## ✨ 주요 기능 (Features)
- **고성능 이미지 분석**: Qwen2.5-VL 모델을 사용하여 높은 정확도로 음식 이미지를 인식합니다.
- **자동 하드웨어 가속**: 실행 환경에 따라 최적의 가속기를 자동으로 선택합니다.
    - **NVIDIA GPU**: CUDA (Float16)
    - **Mac Silicon (M1/M2/M3)**: MPS (Float16)
    - **CPU**: Fallback (Float32)
- **최적화된 전처리**: 이미지 리사이징 및 RGB 변환을 통해 처리 속도를 최적화했습니다.

---

## 🛠️ 설치 및 실행 (Installation & Run)

이 프로젝트는 Python 3.10 이상 버전을 필요로 합니다.

### 1. 프로젝트 클론 (Clone)
```bash
git clone https://github.com/Match-Meal/MM-AI.git
cd MM-AI
```

### 2. 가상환경 설정 (Virtual Environment Setup)
프로젝트의 의존성을 격리하기 위해 가상환경을 사용하는 것을 권장합니다. **운영체제에 맞는 명령어**를 사용하세요.

#### 🍎 Mac / Linux 사용자
```bash
# 가상환경 생성
python3 -m venv venv

# 가상환경 활성화
source venv/bin/activate
```
_(터미널 프롬프트 앞에 `(venv)`가 표시되면 성공입니다.)_

#### 🪟 Windows 사용자
```powershell
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
venv\Scripts\activate
```

### 3. 패키지 설치 (Install Dependencies)
```bash
pip install -r requirements.txt
```

### 4. 서버 실행 (Run Server)
서버를 실행하면 모델 로딩이 시작됩니다. (컴퓨터 사양에 따라 수 초~수 분 소요될 수 있습니다.)
```bash
uvicorn app.main:app --reload
```
- 서버가 정상적으로 실행되면 `http://127.0.0.1:8000` 에서 대기합니다.
- 실행 로그에 `✅ AI 모델 로딩 완료!`가 뜨면 API를 사용할 준비가 된 것입니다.

---

## 🚀 API 사용법 (API Usage)

### 음식 이미지 분석 요청
- **URL**: `POST /vision/analyze`
- **Content-Type**: `multipart/form-data`

#### 요청 파라미터
| 이름 | 타입 | 설명 |
|---|---|---|
| `file` | File | 분석할 음식 이미지 파일 (jpg, png 등) |

#### 응답 예시 (Response)
```json
{
    "candidates": [
        "Spicy Noodles with Shrimp",
        "Korean Stir-Fried Noodles",
        "Sesame Noodles with Vegetables"
    ]
}
```

### 🧪 테스트 방법

#### Option 1: cURL 사용
터미널에서 직접 이미지를 전송하여 테스트할 수 있습니다.
```bash
curl -X POST "http://127.0.0.1:8000/vision/analyze" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/경로/파일이름.jpg"
```

#### Option 2: Python 스크립트 사용
```python
import requests

url = "http://127.0.0.1:8000/vision/analyze"
file_path = "./my_food.jpg"

try:
    with open(file_path, "rb") as f:
        files = {"file": f}
        response = requests.post(url, files=files)
        
    print(response.json())
except Exception as e:
    print(f"Error: {e}")
```

#### Option 3: Swagger UI
브라우저에서 `http://127.0.0.1:8000/docs` 로 접속하면 웹 UI에서 파일을 업로드하고 바로 테스트해볼 수 있습니다.
