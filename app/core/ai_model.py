import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

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
    
    print(f"🔄 AI 모델 로딩 시작... (Target Device: {device.upper()})")
    
    try:
        # 2. device_map 전략 설정
        # CUDA(NVIDIA)는 'auto' 설정이 메모리 관리에 가장 효율적입니다.
        # 반면, MPS(Mac)나 CPU는 'auto' 설정 시 에러가 날 수 있어 수동으로 할당합니다.
        use_device_map = "auto" if device == "cuda" else None
        
        # 3. 모델 로드
        _model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2.5-VL-3B-Instruct",
            torch_dtype=dtype,
            device_map=use_device_map,
        )
        
        # 4. 수동 장치 이동 (MPS/CPU인 경우)
        if not use_device_map:
            _model.to(device)
            
        # 5. 프로세서 로드
        _processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
        
        print("✅ AI 모델 로딩 완료!")
        
    except Exception as e:
        print(f"❌ 모델 로딩 중 치명적인 오류 발생: {e}")
        raise e

def get_model_instance():
    """서비스 계층에서 모델을 호출할 때 사용합니다."""
    if _model is None or _processor is None:
        raise RuntimeError("AI 모델이 아직 로드되지 않았습니다. 서버 실행 로그를 확인하세요.")
    return _model, _processor, _device