from typing import Optional, Dict, Any
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
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
        self._setup_chain()

    def _setup_chain(self):
        # 기본 프롬프트 템플릿 설정
        self.prompt = PromptTemplate(
            template="{prompt}",
            input_variables=["prompt"]
        )
        
        # 변환 체인 생성
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        """
        포스팅을 지정된 감정과 동물 타입의 말투로 변환합니다.
        
        Args:
            content (str): 변환할 원본 텍스트
            emotion (Emotion): 감정 상태
            post_type (PostType): 동물 타입
            
        Returns:
            str: 변환된 텍스트
        """
        try:
            # 프롬프트 생성기 초기화
            prompt_generator = PromptGenerator(
                emotion=emotion.value,  # "일반", "행복" 등
                post_type=post_type.value,  # "고양이", "강아지"
                content=content  # 변환할 원본 텍스트
            )
            
            # 프롬프트 생성
            formatted_prompt = prompt_generator.get_formatted_prompt()
            
            # 체인 실행
            result = await self.chain.arun(
                prompt=formatted_prompt
            )
            
            return result.strip()
            
        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}") 