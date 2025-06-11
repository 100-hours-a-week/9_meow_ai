from ai_server.models.huggingface_model import PostModel
from ai_server.post.post_schemas import Emotion, PostType
from ai_server.post.post_prompt import PostPromptGenerator
from ai_server.models.model_manager import get_model_manager
import logging

# 로깅 설정
logger = logging.getLogger(__name__)

# post 변환 서비스(허깅페이스 모델 사용)
class PostTransformationService:
    def __init__(self, model_name: str = None):
        # ModelManager를 통해 모델 인스턴스 가져오기
        self.model_manager = get_model_manager()
        self.model_name = model_name
        
    # post 변환 서비스 메서드(포스트 변환)
    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        try:
            # ModelManager에서 모델 인스턴스 비동기적으로 가져오기
            model = await self.model_manager.get_post_model_async()
            
            # 1. 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = PostPromptGenerator(
                emotion=emotion.value,
                post_type=post_type.value,
                content=content
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()

            # 2. 모델을 사용하여 텍스트 생성 (모델의 기본 설정값 사용)
            result = await model.generate(
                prompt=formatted_prompt,
                temperature=model.temperature,
                top_p=model.top_p,
                max_new_tokens=model.max_new_tokens,
            )
            
            # 결과가 없으면 원본 내용 반환
            if not result or not result.strip():
                logger.warning("모델이 결과를 생성하지 않았습니다. 원본 내용을 반환합니다.")
                return content
                
            return result.strip()

        except RuntimeError as re:
            logger.error(f"모델 로드 실패: {str(re)}")
            raise Exception(f"Model loading failed: {str(re)}")
        except Exception as e:
            logger.error(f"포스트 변환 실패: {str(e)}")
            raise Exception(f"Failed to transform post: {str(e)}") 