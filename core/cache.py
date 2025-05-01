from typing import Optional, Any
import time
import hashlib
import json
from asyncio import Lock
from collections import OrderedDict

class AsyncLRUCache:
    def __init__(self, capacity: int = 1000, ttl_seconds: int = 3600):
        self.capacity = capacity
        self.ttl_seconds = ttl_seconds
        self.cache = OrderedDict()
        self.lock = Lock()
        self.timestamps = {}

    def _generate_key(self, content: str, emotion: str, post_type: str) -> str:
        """요청 파라미터를 기반으로 캐시 키 생성"""
        key_data = {
            "content": content,
            "emotion": emotion,
            "post_type": post_type
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()

    async def get(self, content: str, emotion: str, post_type: str) -> Optional[str]:
        """캐시된 결과 조회"""
        key = self._generate_key(content, emotion, post_type)
        
        async with self.lock:
            if key not in self.cache:
                return None

            # TTL 체크
            if time.time() - self.timestamps[key] > self.ttl_seconds:
                del self.cache[key]
                del self.timestamps[key]
                return None

            # LRU 업데이트
            self.cache.move_to_end(key)
            return self.cache[key]

    async def put(self, content: str, emotion: str, post_type: str, result: str):
        """결과를 캐시에 저장"""
        key = self._generate_key(content, emotion, post_type)
        
        async with self.lock:
            if key in self.cache:
                self.cache.move_to_end(key)
            else:
                if len(self.cache) >= self.capacity:
                    # 가장 오래된 항목 제거
                    oldest_key, _ = self.cache.popitem(last=False)
                    del self.timestamps[oldest_key]
                
                self.cache[key] = result
                self.timestamps[key] = time.time()

    async def clear_expired(self):
        """만료된 캐시 항목 제거"""
        current_time = time.time()
        async with self.lock:
            expired_keys = [
                key for key, timestamp in self.timestamps.items()
                if current_time - timestamp > self.ttl_seconds
            ]
            for key in expired_keys:
                del self.cache[key]
                del self.timestamps[key] 