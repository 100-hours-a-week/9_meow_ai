from typing import List, Optional, Dict
import time
from asyncio import Lock
from pydantic import BaseModel

class APIKeyStatus(BaseModel):
    key: str                 # 실제 API 키
    label: str               # 출력용 라벨 (예: key_1)
    request_count: int       # 1분 동안 사용된 횟수
    last_used: float = 0     # 마지막 사용 시간

class APIKeyPool:
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 60):
        # API 키와 라벨 매핑 생성
        self.key_to_label: Dict[str, str] = {
            key: f"key_{i+1}" for i, key in enumerate(api_keys)
        }
        
        self.keys: List[APIKeyStatus] = [
            APIKeyStatus(
                key=k,
                label=self.key_to_label[k],
                request_count=0,
                last_used=0.0
            )
            for k in api_keys
        ]
        self.max_requests_per_min = max_requests_per_min
        self.lock = Lock()

    async def get_available_key(self) -> Optional[str]:
        async with self.lock:
            current_time = time.time()
            available_keys = [
                key for key in self.keys
                if key.request_count < self.max_requests_per_min
            ]
            
            if not available_keys:
                return None

            # 가장 적게 사용된 키 선택
            selected_key = min(available_keys, key=lambda k: k.request_count)
            selected_key.request_count += 1
            selected_key.last_used = current_time
            
            print(f"선택된 키 라벨: {selected_key.label}, 요청 카운트: {selected_key.request_count}")
            return selected_key.key

    async def release_key(self, api_key: str):
        async with self.lock:
            for key in self.keys:
                if key.key == api_key:
                    # 키 사용 후 카운트는 유지 (1분 후 reset_counters에서 초기화)
                    break

    async def reset_counters(self):
        async with self.lock:
            for key in self.keys:
                key.request_count = 0
                print(f"키 {key.label} 카운터 리셋됨")
