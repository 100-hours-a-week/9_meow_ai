#!/usr/bin/env python3
"""
vLLM 모델 관리 도구
LoRA와 풀 파인튜닝 모델 간 전환 및 관리
"""

import sys
import os
import argparse
import json
from pathlib import Path
from typing import Dict, Any

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_server.vLLM import (
    VLLMLauncher, 
    ModelDetector, 
    get_vllm_config, 
    switch_model,
    ModelType,
    ModelConfig
)


class ModelManager:
    """모델 관리 클래스"""
    
    def __init__(self):
        self.launcher = VLLMLauncher()
        self.detector = ModelDetector()
        self.config = get_vllm_config()
    
    def list_models(self) -> None:
        """지원되는 모델 목록 출력"""
        print("=== 지원되는 모델 목록 ===")
        print(f"현재 활성 모델: {self.config.active_model}")
        print()
        
        for model_name, model_config in self.config.supported_models.items():
            status = "🟢 활성" if model_name == self.config.active_model else "⚪ 비활성"
            print(f"{status} {model_name}")
            print(f"  타입: {model_config.model_type.value}")
            print(f"  경로: {model_config.model_path}")
            if model_config.base_model_path:
                print(f"  베이스 모델: {model_config.base_model_path}")
            print(f"  서빙명: {model_config.served_model_name}")
            print(f"  GPU 메모리: {model_config.gpu_memory_utilization * 100:.0f}%")
            print()
    
    def switch_model(self, model_name: str) -> bool:
        """모델 전환 및 서버 시작"""
        try:
            print(f"모델을 '{model_name}'으로 전환합니다...")
            
            if model_name not in self.config.supported_models:
                print(f"❌ 지원되지 않는 모델: {model_name}")
                print("지원되는 모델 목록:")
                for name in self.config.supported_models.keys():
                    print(f"  - {name}")
                return False
            
            # 기존 서버가 실행 중이면 먼저 중지
            if self.launcher.process and self.launcher.process.poll() is None:
                print("기존 서버를 중지합니다...")
                self.launcher.stop_server()
            
            # 모델 설정 변경
            from ai_server.vLLM.server.vllm_config import switch_model
            switch_model(model_name)
            
            # 새 설정으로 launcher 업데이트
            self.launcher = VLLMLauncher()
            
            print(f"✅ 모델 설정 변경 완료: {model_name}")
            print("서버를 시작합니다...")
            
            # 서버 시작
            success = self.launcher.start_server()
            
            if success:
                print(f"✅ '{model_name}' 모델로 서버 시작 완료")
                return True
            else:
                print(f"❌ 서버 시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ 모델 전환 중 오류: {e}")
            return False
    
    def start_server(self) -> bool:
        """서버 시작"""
        try:
            print("vLLM 서버를 시작합니다...")
            success = self.launcher.start_server()
            
            if success:
                print("✅ 서버 시작 완료")
                self.show_status()
                return True
            else:
                print("❌ 서버 시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ 서버 시작 중 오류: {e}")
            return False
    
    def stop_server(self) -> bool:
        """서버 중지"""
        try:
            print("vLLM 서버를 중지합니다...")
            success = self.launcher.stop_server()
            
            if success:
                print("✅ 서버 중지 완료")
                return True
            else:
                print("❌ 서버 중지 실패")
                return False
                
        except Exception as e:
            print(f"❌ 서버 중지 중 오류: {e}")
            return False
    
    def restart_server(self) -> bool:
        """서버 재시작"""
        try:
            print("vLLM 서버를 재시작합니다...")
            success = self.launcher.restart_server()
            
            if success:
                print("✅ 서버 재시작 완료")
                self.show_status()
                return True
            else:
                print("❌ 서버 재시작 실패")
                return False
                
        except Exception as e:
            print(f"❌ 서버 재시작 중 오류: {e}")
            return False
    
    def show_status(self) -> None:
        """서버 상태 출력"""
        status = self.launcher.get_server_status()
        
        print("=== vLLM 서버 상태 ===")
        print(f"실행 상태: {'🟢 실행 중' if status['running'] else '🔴 중지'}")
        
        if status['running']:
            print(f"PID: {status['pid']}")
            print(f"주소: http://{status['host']}:{status['port']}")
        
        print(f"활성 모델: {status['active_model']}")
        print(f"모델 타입: {status['model_type']}")
        print(f"서빙 모델명: {status['served_model_name']}")
        print()
    
    def detect_model(self, model_path: str) -> None:
        """모델 타입 감지"""
        try:
            detected_type = self.detector.detect_model_type(model_path)
            
            print(f"감지된 모델 타입: {detected_type.value}")
            
            # 감지 결과에 따른 권장 설정 출력
            if detected_type == ModelType.LORA:
                print("\n📋 LoRA 모델 설정 가이드:")
                print("- base_model_path: 베이스 모델 경로 설정 필요")
                print("- lora_modules: LoRA 어댑터 모듈 설정 필요")
                print("- GPU 메모리 사용률: 0.7 권장")
            elif detected_type == ModelType.FULL_FINETUNED:
                print("\n📋 풀 파인튜닝 모델 설정 가이드:")
                print("- 베이스 모델 경로 불필요")
                print("- GPU 메모리 사용률: 0.9 권장")
                print("- 더 많은 배치 크기 및 시퀀스 길이 설정 가능")
            else:
                print("\n📋 기본 모델 설정 가이드:")
                print("- 표준 설정 사용")
                print("- GPU 메모리 사용률: 0.8 권장")
                
        except Exception as e:
            print(f"❌ 모델 감지 중 오류: {e}")
    
    def export_config(self, output_path: str) -> bool:
        """현재 설정을 파일로 내보내기"""
        try:
            config_dict = {
                "active_model": self.config.active_model,
                "host": self.config.host,
                "port": self.config.port,
                "supported_models": {}
            }
            
            # 모델 설정 직렬화
            for name, model_config in self.config.supported_models.items():
                config_dict["supported_models"][name] = {
                    "model_type": model_config.model_type.value,
                    "model_path": model_config.model_path,
                    "base_model_path": model_config.base_model_path,
                    "lora_modules": model_config.lora_modules,
                    "gpu_memory_utilization": model_config.gpu_memory_utilization,
                    "max_model_len": model_config.max_model_len,
                    "max_num_batched_tokens": model_config.max_num_batched_tokens,
                    "max_num_seqs": model_config.max_num_seqs,
                    "served_model_name": model_config.served_model_name
                }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
            print(f"✅ 설정이 '{output_path}'에 저장되었습니다.")
            return True
            
        except Exception as e:
            print(f"❌ 설정 내보내기 실패: {e}")
            return False


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="vLLM 모델 관리 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  # 모델 목록 확인
  python model_manager.py list
  
  # 모델 전환 및 서버 시작
  python model_manager.py switch haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp16_0619i
  
  # 서버 상태 확인
  python model_manager.py status
  
  # 모델 타입 감지
  python model_manager.py detect haebo/Meow-HyperCLOVAX-1.5B_LoRA_fp16_0619i
  
  # 설정 내보내기
  python model_manager.py export config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='사용 가능한 명령')
    
    # list 명령
    subparsers.add_parser('list', help='지원되는 모델 목록 출력')
    
    # switch 명령
    switch_parser = subparsers.add_parser('switch', help='모델 전환')
    switch_parser.add_argument('model_name', help='전환할 모델 이름')
    switch_parser.add_argument('--no-start', action='store_true', help='서버 시작하지 않음')
    
    # start 명령
    subparsers.add_parser('start', help='서버 시작')
    
    # stop 명령
    subparsers.add_parser('stop', help='서버 중지')
    
    # restart 명령
    subparsers.add_parser('restart', help='서버 재시작')
    
    # status 명령
    subparsers.add_parser('status', help='서버 상태 확인')
    
    # detect 명령
    detect_parser = subparsers.add_parser('detect', help='모델 타입 감지')
    detect_parser.add_argument('model_path', help='감지할 모델 경로')
    
    # export 명령
    export_parser = subparsers.add_parser('export', help='설정 내보내기')
    export_parser.add_argument('output_path', help='출력 파일 경로')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = ModelManager()
    
    try:
        if args.command == 'list':
            manager.list_models()
        
        elif args.command == 'switch':
            success = manager.switch_model(args.model_name)
            if success and not args.no_start:
                manager.start_server()
        
        elif args.command == 'start':
            manager.start_server()
        
        elif args.command == 'stop':
            manager.stop_server()
        
        elif args.command == 'restart':
            manager.restart_server()
        
        elif args.command == 'status':
            manager.show_status()
        
        elif args.command == 'detect':
            manager.detect_model(args.model_path)
        
        elif args.command == 'export':
            manager.export_config(args.output_path)
        
    except KeyboardInterrupt:
        print("\n중단되었습니다.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 