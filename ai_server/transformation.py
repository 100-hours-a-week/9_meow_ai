from typing import Optional, Dict, Any
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from ai_server.schemas import Emotion, PostType
from ai_server.templates import DYNAMIC_PROMPT, ANIMAL_TRAITS

class TransformationService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True,
            google_api_key=api_key  
        )
        self._setup_chain()

    def _setup_chain(self):
        # 동적 프롬프트 템플릿 설정
        self.prompt = PromptTemplate(
            template=DYNAMIC_PROMPT,
            input_variables=["content", "emotion", "animal_type", "suffix", "characteristics"]
        )
        
        # 변환 체인 생성
        self.chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt
        )

    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        """
        포스팅을 지정된 감정과 동물 타입의 말투로 변환합니다.
        """
        try:
            # 동물과 감정에 따른 특성 가져오기
            animal_traits = ANIMAL_TRAITS[post_type.value][emotion.value]
            
            # 체인 실행
            result = await self.chain.arun(
                content=content,
                emotion=emotion.value,
                animal_type=post_type.value,
                suffix=animal_traits["suffix"],
                characteristics=", ".join(animal_traits["characteristics"])
            )
            
            return result.strip()
            
        except KeyError as e:
            raise ValueError(f"Invalid emotion or animal type: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}") 