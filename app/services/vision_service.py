import io
import json
from PIL import Image
from fastapi import UploadFile
from qwen_vl_utils import process_vision_info
from app.core.ai_model import get_model_instance
from app.schemas.dtos import FoodAnalysisResponse

MAX_IMAGE_DIMENSION = 1024

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """이미지 리사이징 및 RGB 변환"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    
    # 너무 큰 이미지는 리사이징 (메모리 절약 및 속도 향상)
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
    
    return image

async def analyze_food_image(file: UploadFile) -> FoodAnalysisResponse:
    # 1. 모델 가져오기
    model, processor, device = get_model_instance()
    
    # 2. 이미지 읽기 및 전처리
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    
    # 3. 프롬프트 작성 (JSON 포맷 강제 + 한국어 전문가 페르소나)
    prompt_text = """
    당신은 한국 음식 전문가입니다. 제공된 이미지를 분석하세요.
    
    질문: 이 음식의 이름은 무엇인가요?
    
    답변 조건:
    1. 반드시 '한국어'로 음식 이름을 답하세요. (예: Kimchi Stew -> 김치찌개)
    2. 가장 가능성이 높은 음식 1개와, 헷갈리는 후보 2개를 포함하세요.
    3. 설명이나 마크다운(```json) 없이 오직 아래 JSON 데이터만 반환하세요.
    
    {
        "best_candidate": "가장 확실한 음식명",
        "candidates": ["후보1", "후보2", "후보3"]
    }
    """
    
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]

    # 4. Qwen 입력 데이터 생성
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    
    # 데이터를 장치(MPS/CUDA/CPU)로 이동
    inputs = inputs.to(device)

    # 5. 추론 (Inference)
    generated_ids = model.generate(**inputs, max_new_tokens=256)
    
    # 6. 결과 디코딩
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    # 7. JSON 파싱 및 반환 (핵심 수정 부분)
    
    # (1) 먼저 마크다운 기호(```json) 등을 제거하여 'clean_text'를 만듭니다.
    clean_text = output_text.replace("```json", "").replace("```", "").strip()

    try:
        # (2) AI가 말을 잘 들어서 JSON 형태일 경우
        data = json.loads(clean_text)
        return FoodAnalysisResponse(**data)
        
    except json.JSONDecodeError:
        # (3) AI가 학습된 본능대로 단답형("닭갈비")만 뱉었을 경우 -> 이게 정답입니다.
        print(f"💡 JSON 파싱 실패 (단답형 응답 감지): {clean_text}")
        
        # 혹시 모를 줄바꿈이나 공백 제거 후 첫 줄만 가져오기
        final_answer = clean_text.split('\n')[0].strip()
        
        return FoodAnalysisResponse(
            best_candidate=final_answer,     # 예: "닭갈비"
            candidates=[final_answer]        # 후보 리스트에도 넣어줌
        )