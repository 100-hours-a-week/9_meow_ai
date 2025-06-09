from typing import Dict, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PostPromptGenerator(BaseModel):
    """포스트용 프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""  
        template = (
            f"### core guidelines.\n"
            f"1. Transform speech without losing the meaning of the original text.\n"
            f"2. Use only one emoticon per sentence.\n"
            f"3. Make a clear distinction between cat and dog responses.\n"
            f"### content:\n{self.content}\n"
            f"### emotion:\n{self.emotion}\n"
            f"### post_type:\n{self.post_type}\n\n"
            f"### transformed_content:"
        )

        return PromptTemplate(
            template=template,
            input_variables=["content", "emotion", "post_type"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(
            content=self.content,
            emotion=self.emotion,
            post_type=self.post_type
        ) 