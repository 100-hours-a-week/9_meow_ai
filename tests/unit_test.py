# unit_test.py

import pytest
from ai_server.core.config import get_settings
from ai_server.util.v1.key_manager import APIKeyPool, initialize_key_pool
from ai_server.model.post_model import PostTransformationService
from ai_server.schemas.post_schemas import PostRequest, Emotion, PostType
# from ai_server.comment.comment_model import CommentTransformationService  # 더 이상 사용하지 않음
# from ai_server.comment.comment_schemas import CommentRequest, CommentType  # 더 이상 사용하지 않음

# test_config.py
def test_get_settings():
    settings = get_settings()
    assert settings is not None, "Settings 인스턴스가 반환되지 않았습니다."
    assert isinstance(settings.GOOGLE_API_KEYS, list), "GOOGLE_API_KEYS는 리스트여야 합니다."

# test_key_manager.py
@pytest.fixture
def key_pool():
    # 테스트를 위한 API 키 풀 초기화
    return initialize_key_pool()

@pytest.mark.asyncio
async def test_get_available_key(key_pool):
    key = await key_pool.get_available_key()
    assert key is not None, "사용 가능한 API 키가 반환되지 않았습니다."

@pytest.mark.asyncio
async def test_get_available_key_none():
    # 빈 API 키 풀로 테스트
    empty_key_pool = APIKeyPool(api_keys=[])
    key = await empty_key_pool.get_available_key()
    assert key is None, "키가 없을 때 None이 반환되어야 합니다."

def test_initialize_key_pool():
    key_pool = initialize_key_pool()
    assert key_pool.get_available_key_count() > 0, "초기화된 키 풀에 적어도 하나의 키가 있어야 합니다."

def test_api_keys_loaded():
    settings = get_settings()
    assert len(settings.GOOGLE_API_KEYS) > 0, "환경 변수에서 API 키가 로드되지 않았습니다."

def test_api_key_pool_properties():
    key_pool = initialize_key_pool()
    assert isinstance(key_pool, APIKeyPool), "반환된 객체가 APIKeyPool 인스턴스가 아닙니다."
    assert hasattr(key_pool, 'api_keys'), "APIKeyPool 인스턴스에 'api_keys' 속성이 없습니다."
    assert hasattr(key_pool, 'lock'), "APIKeyPool 인스턴스에 'lock' 속성이 없습니다."

def test_initialize_key_pool_no_keys(monkeypatch):
    # 환경 변수에서 API 키를 제거하여 테스트
    monkeypatch.setattr(get_settings(), 'GOOGLE_API_KEYS', [])
    with pytest.raises(ValueError, match="GOOGLE_API_KEYS가 비어있습니다."):
        initialize_key_pool()

# test_post_model.py
@pytest.mark.asyncio
async def test_post_transformation_service():
    # PostTransformationService는 vllm_base_url 인자를 받습니다
    service = PostTransformationService(vllm_base_url="http://localhost:8001")
    
    # 테스트 데이터
    request = PostRequest(content="Hello, world!", emotion=Emotion.HAPPY, post_type=PostType.CAT)
    
    # 변환 실행
    transformed_content = await service.transform_post(request.content, request.emotion, request.post_type)
    
    # 결과 검증 (예상된 결과에 따라 수정 필요)
    assert transformed_content is not None, "변환된 콘텐츠가 없습니다."
    assert isinstance(transformed_content, str), "변환된 콘텐츠가 문자열이 아닙니다."

# test_comment_model.py - 더 이상 사용하지 않는 Gemini API 댓글 변환 테스트 제거
# @pytest.mark.asyncio
# async def test_comment_transformation_service():
#     # CommentTransformationService는 api_key 인자를 받습니다
#     settings = get_settings()
#     api_key = settings.GOOGLE_API_KEYS[0]
#     service = CommentTransformationService(api_key=api_key)
#     
#     # 테스트 데이터
#     request = CommentRequest(content="This is a comment.", post_type=CommentType.DOG)
#     
#     # 변환 실행
#     transformed_content = await service.transform_comment(request.content, request.post_type)
#     
#     # 결과 검증 (예상된 결과에 따라 수정 필요)
#     assert transformed_content is not None, "변환된 콘텐츠가 없습니다."
#     assert isinstance(transformed_content, str), "변환된 콘텐츠가 문자열이 아닙니다."