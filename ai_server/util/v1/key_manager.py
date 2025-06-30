from typing import List, Optional
import asyncio
import time
from ai_server.core.config import get_settings

class APIKeyPool:
    """API 키 풀링을 관리하는 클래스

    여러 API 키를 순환하며 사용하여 처리량을 최적화하고,
    비동기 락을 통해 동시성 문제(동시에 요청이 오는 경우)를 해결합니다.
    """
    def __init__(self, api_keys: List[str], max_requests_per_min: int = 15):
        self.api_keys = api_keys  # 사용 가능한 API 키 목록
        self.current_index = 0    # 현재 사용 중인 키의 인덱스
        self.lock = asyncio.Lock()  # 비동기 동시성 제어를 위한 락
        self.key_usage = {key: [] for key in api_keys}  # 키별 사용 시간 기록
        self.max_requests_per_min = max_requests_per_min  # 분당 최대 요청 수

    async def _cleanup_old_usage(self, key: str) -> None:
        """1분 이상 지난 사용 기록을 제거합니다."""
        current_time = time.time()
        self.key_usage[key] = [t for t in self.key_usage[key] if current_time - t < 60]

    async def _is_key_available(self, key: str) -> bool:
        """키의 사용 가능 여부를 확인합니다."""
        await self._cleanup_old_usage(key)
        return len(self.key_usage[key]) < self.max_requests_per_min

    async def get_available_key(self) -> Optional[str]:
        """사용 가능한 API 키를 반환하고 다음 키로 순환합니다.
        
        Returns:
            Optional[str]: 사용 가능한 API 키 또는 None (키가 없는 경우)
        """
        async with self.lock:  # 동시성 제어를 위한 락 획득
            if not self.api_keys:
                return None

            # 현재 키 확인
            key = self.api_keys[self.current_index]
            
            # 키 사용 가능 여부 확인
            if await self._is_key_available(key):
                # 키 사용 기록 추가
                self.key_usage[key].append(time.time())
                # 다음 키로 순환
                self.current_index = (self.current_index + 1) % len(self.api_keys)
                print(f"Selected Key: key_{self.current_index + 1} (usage: {len(self.key_usage[key])}/min)")
                return key
            
            # 키 사용량 초과 시 None 반환
            print(f"Rate limit exceeded: key_{self.current_index + 1} (usage: {len(self.key_usage[key])}/min)")
            return None

    def get_available_key_count(self) -> int:
        """현재 사용 가능한 API 키의 개수를 반환합니다."""
        return len(self.api_keys)

def initialize_key_pool() -> APIKeyPool:
    """환경 변수에서 API 키를 로드하고 키 풀을 초기화합니다.
    
    Settings 클래스를 통해 GOOGLE_API_KEYS를 로드합니다.
    
    Returns:
        APIKeyPool: 초기화된 API 키 풀 인스턴스
    """
    settings = get_settings()
    api_keys = settings.GOOGLE_API_KEYS
    print(f"Loaded Keys: {len(api_keys)}")
    
    if not api_keys:
        raise ValueError("Error: GOOGLE_API_KEYS가 비어있습니다.")
    
    return APIKeyPool(api_keys=api_keys)
