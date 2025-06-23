from typing import Dict, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PostPromptGenerator(BaseModel):
    """포스트용 프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    def create_prompt(self) -> PromptTemplate:
        """HyperCLOVA X 최적화된 간결한 프롬프트 템플릿 생성"""
        
        template = f"""
### Role
당신은 {self.post_type}의 관점에서 인간의 텍스트를 동물의 말투로 재생성하는 작가입니다.

### Context
<transformation_context>
입력 원문: {self.content}
</transformation_context>

### System Instructions
1. 목적: 입력 원문을 {self.post_type}의 {self.emotion}한 말투로 자연스럽게 재생성
2. 톤: {self.emotion}한 감정을 {self.post_type} 특성에 맞게 표현
3. 출력형식: 변환된 텍스트만 반환 (추가 설명 없음)
4. 규칙: 오직 한국어로 변환, 지나친 의성어/의태어 사용 금지

### Task
다음 텍스트를 위 조건에 맞게 재생성해주세요:

<original_text>
{self.content}
</original_text>
"""

        return PromptTemplate(
            input_variables=["content"],
            template=template
        )


    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content)
