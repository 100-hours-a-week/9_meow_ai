from ai_server.schemas.post_schemas import Emotion, PostType
from ai_server.util.post_prompt import PostPromptGenerator
from ai_server.external.vLLM import VLLMAsyncClient, CompletionRequest
from ai_server.core.config import get_inference_config
import logging
import re
# 로깅 설정
logger = logging.getLogger(__name__)

# post 변환 서비스
class PostTransformationService:
    def __init__(self, vllm_base_url: str = "http://localhost:8002"):
        self.vllm_base_url = vllm_base_url
        self.inference_config = get_inference_config()
        
    # post 변환 서비스 메서드
    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        try:
            # 1. 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = PostPromptGenerator(
                emotion=emotion.value,
                post_type=post_type.value,
                content=content
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()

            # 2. VLLMAsyncClient를 사용하여 vLLM 서버에 요청 (최적화된 파라미터)
            async with VLLMAsyncClient(base_url=self.vllm_base_url) as client:
                completion_request = CompletionRequest(
                    prompt=formatted_prompt,
                    max_tokens=self.inference_config.post_max_tokens,
                    temperature=self.inference_config.post_temperature,
                    top_p=self.inference_config.post_top_p,
                    top_k=self.inference_config.post_top_k,
                    stop=self.inference_config.post_stop_tokens
                )
                
                result = await client.completion(completion_request)
                generated_text = result["choices"][0]["text"].strip()

         
                processed_text = self.postprocess(generated_text)
                return processed_text

        except Exception as e:
            logger.error(f"포스트 변환 실패: {str(e)}")
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
        
        # 1. 반복 단어/구(띄어쓰기 단위, 연속 N회 이상 반복되면 자르기)
        #words = text.split()
        #seen = {}
        #for idx, word in enumerate(words):
        #    seen[word] = seen.get(word, 0) + 1
        #    if seen[word] > max_repeat:
        #        text = ' '.join(words[:idx])
        #        break

        # 2. 너무 긴 답변은 문장 경계(마침표, 느낌표, 물음표) 기준으로 자르기
        #if len(text) > max_len:
        #    cut = text[:max_len]
        #    m = re.search(r'[.!?](?=[^.!?]*$)', cut)
        #    if m:
        #        text = cut[:m.end()]
        #    else:
        #        text = cut

        # 3. 남은 중복 공백 및 앞뒤 공백 정리
        #text = re.sub(r'\s+', ' ', text)
        #return text.strip()
        return text
 