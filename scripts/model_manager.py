#!/usr/bin/env python3
"""
vLLM 서버 관리 도구 - 간소화 버전
"""

import sys
import argparse
from pathlib import Path

# 프로젝트 루트를 Python 경로에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_server.external.vLLM import VLLMLauncher, get_vllm_config


class SimpleModelManager:
    """간소화된 모델 관리 클래스"""
    
    def __init__(self):
        self.launcher = VLLMLauncher()
        self.config = get_vllm_config()
    
    def show_current_model(self) -> None:
        """현재 모델 정보 출력"""
        print("=== 현재 모델 설정 ===")
        print(f"모델 경로: {self.config.model_path}")
        print(f"서빙 모델명: {self.config.served_model_name}")
        print(f"서버 주소: {self.config.host}:{self.config.port}")
        print()
    
    def start_server(self) -> bool:
        """서버 시작"""
        try:
            print("vLLM 서버를 시작합니다...")
            print(f"모델: {self.config.model_path}")
            
            success = self.launcher.start_server()
            
            if success:
                print("서버 시작 완료")
                self.show_status()
                return True
            else:
                print("서버 시작 실패")
                return False
                
        except Exception as e:
            print(f"서버 시작 중 오류: {e}")
            return False
    
    def stop_server(self) -> bool:
        """서버 중지"""
        try:
            print("vLLM 서버를 중지합니다...")
            success = self.launcher.stop_server()
            
            if success:
                print("서버 중지 완료")
                return True
            else:
                print("서버 중지 실패")
                return False
                
        except Exception as e:
            print(f"서버 중지 중 오류: {e}")
            return False
    
    def restart_server(self) -> bool:
        """서버 재시작"""
        try:
            print("vLLM 서버를 재시작합니다...")
            success = self.launcher.restart_server()
            
            if success:
                print("서버 재시작 완료")
                self.show_status()
                return True
            else:
                print("서버 재시작 실패")
                return False
                
        except Exception as e:
            print(f"서버 재시작 중 오류: {e}")
            return False
    
    def show_status(self) -> None:
        """서버 상태 출력"""
        is_running = self.launcher.is_running()
        
        print("=== vLLM 서버 상태 ===")
        print(f"실행 상태: {'실행 중' if is_running else '중지'}")
        
        if is_running:
            print(f"주소: http://{self.config.host}:{self.config.port}")
            print(f"모델: {self.config.model_path}")
            print(f"서빙명: {self.config.served_model_name}")
        print()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="vLLM 서버 관리 도구")
    parser.add_argument("action", choices=["start", "stop", "restart", "status", "info"], 
                       help="실행할 액션")
    
    args = parser.parse_args()
    manager = SimpleModelManager()
    
    success = True
    
    if args.action == "start":
        success = manager.start_server()
    elif args.action == "stop":
        success = manager.stop_server()
    elif args.action == "restart":
        success = manager.restart_server()
    elif args.action == "status":
        manager.show_status()
    elif args.action == "info":
        manager.show_current_model()
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 