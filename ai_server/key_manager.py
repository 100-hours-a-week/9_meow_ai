from typing import List, Optional
import asyncio
import ast
import os
from dotenv import load_dotenv

class APIKeyPool:
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 5):
        self.api_keys = api_keys
        self.max_requests_per_min = max_requests_per_min
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def get_available_key(self) -> Optional[str]:
        async with self.lock:
            if not self.api_keys:
                return None
                
            # 현재 인덱스의 키 반환
            key = self.api_keys[self.current_index]
            
            # 다음 키로 인덱스 이동
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            
            print(f"선택된 키: key_{self.current_index + 1}")
            return key

    async def release_key(self, key: str):
        # 순차적 사용에서는 release가 필요 없음
        pass

    async def reset_counters(self):
        # 순차적 사용에서는 reset이 필요 없음
        pass

    def get_available_key_count(self) -> int:
        return len(self.api_keys)

def initialize_key_pool() -> APIKeyPool:
    """API 키 풀 초기화 및 설정"""
    load_dotenv()
    api_keys = []
    single_key = os.getenv("GOOGLE_API_KEYS")
    multiple_keys = os.getenv("GOOGLE_API_KEYS_list")

    if multiple_keys:
        try:
            api_keys = ast.literal_eval(multiple_keys)
            if not isinstance(api_keys, list):
                print("Warning: GOOGLE_API_KEYS_list must be a list")
                api_keys = []
            else:
                print(f"Running in multi-key mode with {len(api_keys)} keys")
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse GOOGLE_API_KEYS_list: {e}")
            api_keys = []
    elif single_key:
        try:
            # 단일 키도 리스트로 변환
            api_keys = ast.literal_eval(single_key)
            if not isinstance(api_keys, list):
                api_keys = [api_keys]  # 단일 키를 리스트로 변환
            print(f"Running in single-key mode with {len(api_keys)} keys")
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse GOOGLE_API_KEYS: {e}")
            api_keys = []
    else:
        print("Warning: No API keys found. Please set either GOOGLE_API_KEYS or GOOGLE_API_KEYS_list")
        api_keys = []

    if not api_keys:
        print("Warning: No valid API keys found")

    return APIKeyPool(api_keys=api_keys, max_requests_per_min=5) 