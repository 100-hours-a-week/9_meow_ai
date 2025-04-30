from typing import Optional, Dict, Any, List
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from ai_server.schemas import Emotion, PostType
from ai_server.templates import DYNAMIC_PROMPT, ANIMAL_TRAITS
from ai_server.key_pool import APIKeyPool
from ai_server.cache import AsyncLRUCache
from ai_server.batch_processor import BatchProcessor

class TransformationService:
    def __init__(self, api_keys: List[str]):
        self.key_pool = APIKeyPool(api_keys)
        self.cache = AsyncLRUCache()
        self.batch_processor = BatchProcessor(self)
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

    async def _get_llm(self):
        """사용 가능한 API 키로 LLM 인스턴스 생성"""
        api_key = await self.key_pool.get_available_key()
        if not api_key:
            raise ValueError("No API key available")
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True,
            google_api_key=api_key
        )
        return llm, api_key

    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        """
        포스팅을 지정된 감정과 동물 타입의 말투로 변환합니다.
        캐시 확인 -> 배치 처리 -> API 호출 순으로 처리
        """
        try:
            # 1. 캐시 확인
            cached_result = await self.cache.get(content, emotion.value, post_type.value)
            if cached_result:
                return cached_result

            # 2. 배치 처리에 추가
            result = await self.batch_processor.add_request({
                "content": content,
                "emotion": emotion,
                "post_type": post_type
            })

            # 3. 결과 캐싱
            await self.cache.put(content, emotion.value, post_type.value, result)
            
            return result
            
        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}")

    async def _transform_single(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        """단일 요청 처리 (배치 처리에서 호출)"""
        try:
            llm, api_key = await self._get_llm()
            
            try:
                chain = LLMChain(llm=llm, prompt=self.prompt)
                animal_traits = ANIMAL_TRAITS[post_type.value][emotion.value]
                
                result = await chain.arun(
                    content=content,
                    emotion=emotion.value,
                    animal_type=post_type.value,
                    suffix=animal_traits["suffix"],
                    characteristics=", ".join(animal_traits["characteristics"])
                )
                
                return result.strip()
                
            finally:
                await self.key_pool.release_key(api_key)
                
        except KeyError as e:
            raise ValueError(f"Invalid emotion or animal type: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}")

    async def cleanup(self):
        """주기적인 정리 작업"""
        await self.cache.clear_expired()
        await self.key_pool.reset_counters() 