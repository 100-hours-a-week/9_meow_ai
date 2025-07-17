from ai_server.schemas.converter_schemas import CommentType, CommentEmotion
from ai_server.util.comment_prompt import CommentPromptGenerator
from ai_server.external.vLLM import VLLMAsyncClient, CompletionRequest
from ai_server.core.config import get_inference_config
import logging
import re

# 로깅 설정
logger = logging.getLogger(__name__)

# comment 변환 서비스
class CommentTransformationService:
    def __init__(self, vllm_base_url: str = "http://localhost:8002"):
        self.vllm_base_url = vllm_base_url
        self.inference_config = get_inference_config()
        
    # comment 변환 서비스 메서드 (vLLM 추론 로직만 사용)
    async def transform_comment(self, content: str, emotion: CommentEmotion, post_type: CommentType) -> str:
        try:
            # 감정은 무조건 normal로 고정
            fixed_emotion = "normal"
            
            # 1. 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = CommentPromptGenerator(
                emotion=fixed_emotion,
                post_type=post_type.value,
                content=content
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()

            # 2. VLLMAsyncClient를 사용하여 vLLM 서버에 요청
            async with VLLMAsyncClient(base_url=self.vllm_base_url) as client:
                completion_request = CompletionRequest(
                    prompt=formatted_prompt,
                    max_tokens=self.inference_config.comment_max_tokens,
                    temperature=self.inference_config.comment_temperature,
                    top_p=self.inference_config.comment_top_p,
                    top_k=self.inference_config.comment_top_k,
                    stop=self.inference_config.comment_stop_tokens
                )
                
                result = await client.completion(completion_request)
                generated_text = result["choices"][0]["text"].strip()

                processed_text = self.postprocess(generated_text)
                return processed_text

        except Exception as e:
            logger.error(f"댓글 변환 실패: {str(e)}")
            # 오류 시 원본 반환
            return content
    
    def postprocess(self, text: str, max_repeat: int = 3, max_len: int = 180) -> str:
        """
        - 모델 특수 토큰 및 불필요한 출력 제거
        - 반복 단어/구절 간단 제거
        - 출력이 너무 길 경우 문장 경계 기준 자르기
        - 남은 불필요 공백 정리
        """
        # 0. 모델 특수 토큰 및 불필요한 출력 제거
        if "Output:" in text:
            text = text.split("Output:", 1)[-1]
        text = text.strip().split('\n')[0]
        text = re.sub(r'</s>', '', text)
        text = re.sub(r'<\|endof.*?\|>', '', text, flags=re.IGNORECASE)
        text = re.sub(r'<\|.*?\|>', '', text)
        text = text.replace('<s>', '')
        text = text.strip()
        
        return text 