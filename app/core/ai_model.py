import os
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel  # ★ 추가된 라이브러리

# 전역 변수 (싱글톤 패턴)
_model = None
_processor = None
_device = "cpu"

def get_device_and_dtype():
    """
    현재 실행 중인 컴퓨터의 하드웨어를 감지하여 
    최적의 장치(device)와 데이터 타입(dtype)을 반환합니다.
    """
    # 1순위: NVIDIA GPU (CUDA) - Windows/Linux
    if torch.cuda.is_available():
        print("✅ 하드웨어 감지: NVIDIA GPU (CUDA)")
        return "cuda", torch.float16
    
    # 2순위: Apple Silicon (MPS) - Mac M1/M2/M3
    elif torch.backends.mps.is_available():
        print("✅ 하드웨어 감지: Apple Silicon (MPS)")
        return "mps", torch.float16
    
    # 3순위: CPU (Fallback) - GPU가 없는 서버 등
    else:
        print("⚠️ 하드웨어 감지: GPU 없음 (CPU 사용)")
        print("   -> CPU는 속도가 느리며, 호환성을 위해 FP32를 사용합니다.")
        return "cpu", torch.float32

def load_model():
    """서버 시작 시 AI 모델을 로드합니다."""
    global _model, _processor, _device
    
    # 1. 환경 감지
    device, dtype = get_device_and_dtype()
    _device = device
    
    print(f"🔄 기본 Qwen 모델 로딩 중... (Target: {device.upper()})")
    
    try:
        # 1. 기본 모델 로드 (인터넷에서 다운로드 or 캐시 사용)
        base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=dtype,
            # Mac(MPS)에서는 device_map="auto"가 불안정할 수 있어 수동 이동 추천
            device_map=None 
        )
        
        # 2. ★ 핵심! 우리가 만든 어댑터(LoRA) 장착
        # 경로: 프로젝트 루트 기준 (./models/food_adapter)
        adapter_path = os.path.join(os.getcwd(), "models", "food_adapter_v1.0")
        
        if os.path.exists(adapter_path):
            print(f"🧩 학습된 어댑터 합체 중... ({adapter_path})")
            _model = PeftModel.from_pretrained(
                base_model, 
                adapter_path,
                torch_dtype=dtype
            )
        else:
            print(f"⚠️ 경고: 어댑터 폴더를 찾을 수 없습니다! ({adapter_path})")
            print("   -> 기본 모델로만 동작합니다.")
            _model = base_model

        # 3. 모델을 장치(MPS/GPU)로 이동
        _model.to(device)
        _model.eval() # 추론 모드 전환

        # 4. 프로세서 로드
        _processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("✅ AI 모델(LoRA) 로딩 완료! 준비 끝!")
        
    except Exception as e:
        print(f"❌ 모델 로딩 실패: {e}")
        raise e

def get_model_instance():
    """서비스 계층에서 모델을 호출할 때 사용합니다."""
    if _model is None or _processor is None:
        raise RuntimeError("AI 모델이 아직 로드되지 않았습니다. 서버 실행 로그를 확인하세요.")
    return _model, _processor, _device