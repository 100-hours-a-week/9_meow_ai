from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from ai_server.post.post_schemas import Emotion, PostType
from ai_server.post.post_prompt import PostPromptGenerator

#post 변환 서비스(제미나이 플래시 모델 사용)
class PostTransformationService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.3,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
    #post 변환 서비스 메서드(포스트 변환)
    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        try:
            # 1. 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = PostPromptGenerator(
                emotion=emotion.value,
                post_type=post_type.value,
                content=content
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()  # -> str 반환

            # 2. LLM에 문자열 직접 전달 (체인 없이)
            result = await self.llm.ainvoke(formatted_prompt)

            # 3. ChatModel의 출력은 message 객체이므로 .content 접근 필요
            return result.content.strip()

        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}") 