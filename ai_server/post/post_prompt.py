from typing import Dict, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
    
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

        # CLOVAX 형식 프롬프트
        template = (
            "### Instruction:\n다음 문장을 감정과 유형에 맞게 말투를 바꿔주세요.\n\n"
            f"### Input:\n문장: {self.content}\n감정: {self.emotion}\n유형: {self.post_type}\n\n"
            "### transformed_content:"
        )

        return PromptTemplate(
            input_variables=["content", "emotion", "post_type"],
            template=template
        )

    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(
            content=self.content,
            emotion=self.emotion,
            post_type=self.post_type
        )
