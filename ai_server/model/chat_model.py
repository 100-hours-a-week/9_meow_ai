from ai_server.schemas.chat_schemas import ChatAnimalType
from ai_server.model.cat import cat_converter
from ai_server.model.dog import dog_converter
from ai_server.model.hamster import hamster_converter
from ai_server.model.monkey import monkey_converter
from ai_server.model.raccoon import raccoon_converter
import logging
from typing import Dict, Callable

# 로깅 설정
logger = logging.getLogger(__name__)

# chat 변환 서비스
class ChatTransformationService:
    def __init__(self):
        # 규칙 기반 변환 함수 매핑
        self.rule_based_converters: Dict[str, Callable[[str], str]] = {
            "cat": cat_converter,
            "dog": dog_converter,
            "hamster": hamster_converter,
            "monkey": monkey_converter,
            "raccoon": raccoon_converter
        }
        
    def transform_chat(self, text: str, post_type: ChatAnimalType) -> str:
        """채팅 텍스트를 규칙 기반으로 변환"""
        try:
            converter = self.rule_based_converters.get(post_type.value)
            if converter:
                return converter(text)
            else:
                logger.warning(f"지원하지 않는 post_type: {post_type.value}")
                return text
        except Exception as e:
            logger.error(f"채팅 변환 실패: {str(e)}")
            return text 