from typing import Optional, Dict, Any
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import google.generativeai as genai
from ai_server.schemas import Emotion, PostType
from ai_server.templates import PROMPT_TEMPLATES

class TransformationService:
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            temperature=0.7,
            convert_system_message_to_human=True
        )
        self._setup_prompts()

    def _setup_prompts(self):
        self.prompt_templates = {
            Emotion.NORMAL: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_normal"]),
            Emotion.HAPPY: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_happy"]),
            Emotion.CURIOUS: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_curious"]),
            Emotion.SAD: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_sad"]),
            Emotion.GRUMPY: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_grumpy"]),
            Emotion.ANGRY: PromptTemplate.from_template(PROMPT_TEMPLATES["cat_angry"]),
        }

    async def transform_post(self, content: str, emotion: Emotion, post_type: PostType) -> str:
        """
        포스팅을 고양이 말투로 변환합니다.
        """
        try:
            # Get appropriate prompt template based on emotion
            prompt = self.prompt_templates[emotion]
            
            # Create chain
            chain = LLMChain(llm=self.llm, prompt=prompt)
            
            # Run chain with content parameter
            result = await chain.arun(content=content)
            
            # Return the transformed content
            return result.strip()
            
        except KeyError as e:
            raise ValueError(f"Invalid emotion: {emotion}")
        except Exception as e:
            raise Exception(f"Failed to transform post: {str(e)}") 