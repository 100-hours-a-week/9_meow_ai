#!/usr/bin/env python3
"""
vLLM 서버 시작 스크립트
HyperCLOVAX-1.5B_LoRA_fp16 모델을 위한 vLLM 서버 실행
"""

import sys
import os
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_server.vllm_server.vllm_launcher import VLLMLauncher
from ai_server.vllm_server.vllm_config import update_vllm_config


def main():
    parser = argparse.ArgumentParser(description="vLLM 서버 시작")
    parser.add_argument("--model-path", 
                       default="./models/haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0527",
                       help="모델 파일 경로")
    parser.add_argument("--port", type=int, default=8001, help="서버 포트")
    parser.add_argument("--host", default="0.0.0.0", help="서버 호스트")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                       help="GPU 메모리 사용률 (0.0-1.0)")
    parser.add_argument("--max-model-len", type=int, default=4096,
                       help="최대 모델 길이")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                       help="텐서 병렬 처리 크기")
    
    args = parser.parse_args()
    
    # 설정 업데이트
    update_vllm_config(
        model_path=args.model_path,
        port=args.port,
        host=args.host,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        tensor_parallel_size=args.tensor_parallel_size
    )
    
    # vLLM 서버 시작
    launcher = VLLMLauncher()
    
    print("=" * 60)
    print("vLLM 서버 시작 중...")
    print(f"모델: {args.model_path}")
    print(f"서버: {args.host}:{args.port}")
    print(f"GPU 메모리 사용률: {args.gpu_memory_utilization * 100:.1f}%")
    print("=" * 60)
    
    try:
        success = launcher.start_server()
        if success:
            print("vLLM 서버가 성공적으로 시작되었습니다.")
            print(f"API 엔드포인트: http://{args.host}:{args.port}")
            print(f"OpenAI 호환 API: http://{args.host}:{args.port}/v1")
            print("서버를 중지하려면 Ctrl+C를 누르세요.")
            
            # 서버 실행 유지
            try:
                import signal
                import time
                
                def signal_handler(signum, frame):
                    print("\n🛑 서버 중지 신호를 받았습니다...")
                    launcher.stop_server()
                    sys.exit(0)
                
                signal.signal(signal.SIGINT, signal_handler)
                signal.signal(signal.SIGTERM, signal_handler)
                
                while True:
                    time.sleep(1)
                    
            except KeyboardInterrupt:
                print("\n서버를 중지합니다...")
                launcher.stop_server()
        else:
            print("vLLM 서버 시작에 실패했습니다.")
            sys.exit(1)
            
    except Exception as e:
        print(f"오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 