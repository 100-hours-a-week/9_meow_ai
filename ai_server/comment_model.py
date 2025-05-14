from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from ai_server.comment_prompt import CommentPromptGenerator
from ai_server.schemas import CommentType

class CommentTransformationService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.5,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
            
    async def transform_comment(self, content: str, post_type: CommentType) -> str:
        try:

            
            # 프롬프트 생성기 통해 텍스트 프롬프트 생성
            prompt_generator = CommentPromptGenerator(
                post_type=post_type.value,
                content=content
            )
            formatted_prompt = prompt_generator.get_formatted_prompt()
            
            # LLM에 문자열 직접 전달
            result = await self.llm.ainvoke(formatted_prompt)
            
            # 결과 반환
            return result.content.strip()
            
        except Exception as e:
            raise Exception(f"Failed to transform comment: {str(e)}") 