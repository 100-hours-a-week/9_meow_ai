#!/usr/bin/env python3
"""
vLLM 서버 시작 스크립트
포스트 문장 생성을 위한 간소화된 vLLM 서버 실행
"""

import sys
import os
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import logging
import time

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_server.vLLM import VLLMLauncher, update_vllm_config


def download_huggingface_model(model_name: str, local_dir: str) -> str:
    """허깅페이스에서 모델 다운로드"""
    try:
        logger.info(f"📥 허깅페이스에서 모델 다운로드 중: {model_name}")
        local_path = snapshot_download(
            repo_id=model_name,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            resume_download=True
        )
        logger.info(f"✅ 모델 다운로드 완료: {local_path}")
        return local_path
    except Exception as e:
        logger.error(f"❌ 모델 다운로드 실패: {e}")
        raise


def setup_lora_models(base_model_name: str, lora_adapter_name: str, models_dir: str) -> tuple[str, str]:
    """LoRA 모델 설정 - 베이스 모델과 어댑터 다운로드"""
    models_path = Path(models_dir)
    models_path.mkdir(parents=True, exist_ok=True)
    
    # 베이스 모델 다운로드
    base_model_local = models_path / "base_model" 
    if not base_model_local.exists():
        logger.info(f"🔽 베이스 모델 다운로드: {base_model_name}")
        base_model_path = download_huggingface_model(base_model_name, str(base_model_local))
    else:
        logger.info(f"📁 기존 베이스 모델 사용: {base_model_local}")
        base_model_path = str(base_model_local)
    
    # 베이스 모델 검증
    if not validate_model_files(base_model_path, "base"):
        raise RuntimeError(f"베이스 모델 검증 실패: {base_model_path}")
    
    # LoRA 어댑터 다운로드  
    lora_adapter_local = models_path / "lora_adapter"
    if not lora_adapter_local.exists():
        logger.info(f"🔽 LoRA 어댑터 다운로드: {lora_adapter_name}")
        lora_adapter_path = download_huggingface_model(lora_adapter_name, str(lora_adapter_local))
    else:
        logger.info(f"📁 기존 LoRA 어댑터 사용: {lora_adapter_local}")
        lora_adapter_path = str(lora_adapter_local)
    
    # LoRA 어댑터 검증
    if not validate_model_files(lora_adapter_path, "lora"):
        raise RuntimeError(f"LoRA 어댑터 검증 실패: {lora_adapter_path}")
    
    return base_model_path, lora_adapter_path


def main():
    parser = argparse.ArgumentParser(description="포스트 생성용 vLLM 서버 시작")
    
    # 기본 모델 설정 (포스트 생성 최적화)
    parser.add_argument("--base-model-name", 
                       default="naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                       help="허깅페이스 베이스 모델 이름")
    parser.add_argument("--lora-adapter-name", 
                       default="haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0527",
                       help="허깅페이스 LoRA 어댑터 이름")
    parser.add_argument("--models-dir", 
                       default="./models",
                       help="모델 저장 디렉토리")
    
    # 서버 설정 (포스트 생성에 필요한 최소 옵션만)
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="GPU 메모리 사용률")
    
    args = parser.parse_args()
    
    try:
        # LoRA 모드로 포스트 생성 서버 시작
        logger.info("🚀 포스트 생성용 LoRA 모델 서버를 시작합니다...")
        
        base_model_path, lora_adapter_path = setup_lora_models(
            args.base_model_name,
            args.lora_adapter_name, 
            args.models_dir
        )
        
        # 포스트 생성 최적화 설정
        update_vllm_config(
            enable_lora=True,
            base_model_path=base_model_path,
            lora_modules=[f"lora={lora_adapter_path}"],
            model_path=base_model_path,
            port=args.port,
            host=args.host,
            gpu_memory_utilization=args.gpu_memory_utilization,
        )
        
        print("=" * 80)
        print("🎯 포스트 생성용 vLLM 서버 시작 중...")
        print(f"📦 베이스 모델: {args.base_model_name}")
        print(f"🔧 LoRA 어댑터: {args.lora_adapter_name}")
        print(f"🌐 서버: {args.host}:{args.port}")
        print(f"💾 GPU 메모리 사용률: {args.gpu_memory_utilization * 100:.1f}%")
        print("=" * 80)
        
        # vLLM 서버 시작
        launcher = VLLMLauncher()
        
        success = launcher.start_server()
        if success:
            print("✅ 포스트 생성용 vLLM 서버가 성공적으로 시작되었습니다!")
            print(f"🔗 API 엔드포인트: http://{args.host}:{args.port}")
            print(f"🔗 OpenAI 호환 API: http://{args.host}:{args.port}/v1")
            print("⏹️  서버를 중지하려면 Ctrl+C를 누르세요.")
            
            # 서버 실행 유지
            try:
                import signal
                
                def signal_handler(signum, frame):
                    print("\n🛑 서버 중지 신호를 받았습니다...")
                    launcher.stop_server()
                    sys.exit(0)
                
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n⏹️  서버를 중지합니다...")
                launcher.stop_server()
        else:
            print("❌ vLLM 서버 시작에 실패했습니다.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 