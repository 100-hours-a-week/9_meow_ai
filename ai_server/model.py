from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from ai_server.schemas import Emotion, PostType
from ai_server.prompt import PromptGenerator

class TransformationService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )

    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        try:
            # 1. 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = PromptGenerator(
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
            
    async def transform_comment(self, text: str, post_type: str) -> str:
        try:
            # post_type 파라미터 변환 (고양이 -> cat, 강아지 -> dog)
            animal_type = "cat" if post_type == "고양이" else "dog"
            
            # 기본 감정 사용 (normal)
            emotion = "normal"
            
            # 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = PromptGenerator(
                emotion=emotion,
                post_type=animal_type,
                content=text
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()
            
            # LLM에 문자열 직접 전달
            result = await self.llm.ainvoke(formatted_prompt)
            
            # 결과 반환
            return result.content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to transform comment: {str(e)}")