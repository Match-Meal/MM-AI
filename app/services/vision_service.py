# 이미지 리사이징(전처리) + 프롬프트 작성 + AI 추론 로직

import io
import json
from PIL import Image
from fastapi import UploadFile
from qwen_vl_utils import process_vision_info
from app.core.ai_model import get_model_instance
from app.schemas.dtos import FoodAnalysisResponse

MAX_IMAGE_DIMENSION = 1280

def preprocess_image(image_bytes: bytes) -> Image.Image:
    """이미지 리사이징 및 RGB 변환"""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    width, height = image.size
    
    if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
    
    return image

async def analyze_food_image(file: UploadFile) -> FoodAnalysisResponse:
    # 1. 모델 가져오기
    model, processor, device = get_model_instance()
    
    # 2. 이미지 읽기 및 전처리
    image_bytes = await file.read()
    image = preprocess_image(image_bytes)
    
    # 3. 프롬프트 작성 (JSON 포맷 강제)
    prompt_text = """
    Analyze this food image. 
    Identify the Korean food name based on visual features.
    Provide top 3 likely candidates.
    Return ONLY a JSON object with this format (no markdown, no extra text):
    {
        "candidates": ["1st choice", "2nd choice", "3rd choice"],
        "best_candidate": "1st choice"
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

    # 7. JSON 파싱 및 반환
    try:
        clean_text = output_text.replace("```json", "").replace("```", "").strip()
        data = json.loads(clean_text)
        return FoodAnalysisResponse(**data)
    except json.JSONDecodeError:
        # 파싱 실패 시 원본 텍스트라도 반환 (디버깅용)
        print(f"JSON Parsing Error. Raw output: {output_text}")
        return FoodAnalysisResponse(
            candidates=[],
            best_candidate="분석 실패 (형식 오류)"
        )