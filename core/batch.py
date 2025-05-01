from typing import List, Dict, Any
import asyncio
from datetime import datetime
from pydantic import BaseModel
from ai_server.schemas import PostRequest, PostResponse

class BatchRequest(BaseModel):
    requests: List[PostRequest]
    batch_id: str

class BatchProcessor:
    def __init__(self, transformation_service, max_batch_size: int = 10, max_wait_time: float = 2.0):
        self.transformation_service = transformation_service
        self.max_batch_size = max_batch_size
        self.max_wait_time = max_wait_time
        self.current_batch: List[Dict[str, Any]] = []
        self.lock = asyncio.Lock()
        self.processing = False

    async def add_request(self, request: PostRequest) -> str:
        """요청을 배치에 추가하고 배치 ID 반환"""
        batch_item = {
            "request": request,
            "future": asyncio.Future(),
            "added_time": datetime.now()
        }
        
        async with self.lock:
            self.current_batch.append(batch_item)
            
            # 배치 크기가 최대에 도달하면 즉시 처리
            if len(self.current_batch) >= self.max_batch_size:
                await self._process_batch()
            # 아니면 배치 처리 태스크 예약
            elif not self.processing:
                self.processing = True
                asyncio.create_task(self._delayed_process())
        
        # 결과 대기
        return await batch_item["future"]

    async def _delayed_process(self):
        """최대 대기 시간 후 배치 처리"""
        await asyncio.sleep(self.max_wait_time)
        async with self.lock:
            if self.current_batch:
                await self._process_batch()
            self.processing = False

    async def _process_batch(self):
        """현재 배치의 모든 요청 처리"""
        batch = self.current_batch
        self.current_batch = []
        
        # 병렬로 모든 요청 처리
        tasks = []
        for item in batch:
            task = asyncio.create_task(
                self.transformation_service.transform_post(
                    content=item["request"].content,
                    emotion=item["request"].emotion,
                    post_type=item["request"].post_type
                )
            )
            tasks.append((task, item["future"]))
        
        # 결과 수집 및 전달
        for task, future in tasks:
            try:
                result = await task
                future.set_result(result)
            except Exception as e:
                future.set_exception(e) 