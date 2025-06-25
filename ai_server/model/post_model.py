from ai_server.schemas.post_schemas import Emotion, PostType
from ai_server.util.post_prompt import PostPromptGenerator
from ai_server.external.vLLM import VLLMAsyncClient, CompletionRequest
from ai_server.core.config import get_inference_config
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# post 변환 서비스
class PostTransformationService:
    def __init__(self, vllm_base_url: str = "http://localhost:8001"):
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
                
                # 후처리
                processed_text = self._postprocess(generated_text, content)
                return processed_text

        except Exception as e:
            logger.error(f"포스트 변환 실패: {str(e)}")
            # 오류 시 원본 반환
            return content
    
    def _postprocess(self, text: str, original_content: str = "") -> str:
        """텍스트 후처리"""
        import re

        EMOJI_PATTERN = (
            "["
            "\U0001F600-\U0001F64F"  # Emoticons
            "\U0001F300-\U0001F5FF"  # Misc Symbols and Pictographs
            "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
            "\U0001F1E0-\U0001F1FF"  # Regional Indicator Symbols
            "\U00002700-\U000027BF"  # Dingbats
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U00002600-\U000026FF"  # Misc Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U0001F700-\U0001F77F"  # Alchemical Symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U00002300-\U000023FF"  # Misc Technical
            "]"
        )

        # content 응답 포맷 제거
        text = re.sub(r"### transformed_content:\s*", "", text).strip()
        text = re.sub(r"### Output:\s*", "", text).strip()
        
        # "Input:" 패턴 제거 (대소문자 구분 없이)
        text = re.sub(r"Input:\s*", "", text, flags=re.IGNORECASE).strip()

        # 줄바꿈 및 백슬래시 제거
        text = re.sub(r'(\\r\\n|\\r|\\n|\r|\n)', '', text)

        # 해시태그 제거
        text = re.sub(r'#\S+', '', text)

        # 이모지 연속 제거
        text = re.sub(f"({EMOJI_PATTERN}){EMOJI_PATTERN}+", r"\1", text)

        # 이모지 2개 초과 시 앞의 2개만 남기고 제거
        emojis = re.findall(EMOJI_PATTERN, text)
        if len(emojis) > 2:
            keep = emojis[:2]
            text = re.sub(EMOJI_PATTERN, '', text) + ''.join(keep)

        # 기호나 감탄사 반복 압축
        text = re.sub(r'([!?\.💢❤⭐✨🐾…]{1})( \1|\1){2,}', r'\1\1', text)

        # 특수문자/비정상 문자 제거
        text = re.sub(r"[️‹›／]", '', text)
        text = re.sub(r'[\u200b\u200c\u200d\u2060\ufeff\u202a-\u202e\u00ad\u034f]', '', text)

        # 너무 짧거나 의미 없는 문장 제거
        if len(text) < 5 or re.fullmatch(r'[\W\d\s]+', text):
            return "[출력 오류] 결과 생성이 실패했어요."

        # 길이 제한
        if original_content:
            max_chars = min(len(text), 400)
            threshold = min(int(len(original_content) * 3.0), max_chars)
            snippet = text[:threshold]
            
            # 자연스러운 문장 끝 찾기
            m = re.search(r'[\.!?](?=[^\.!?]*$)', snippet)
            if m:
                text = snippet[:m.end()]
            else:
                text = snippet.rstrip()

        return re.sub(r'\s+', ' ', text).strip() 