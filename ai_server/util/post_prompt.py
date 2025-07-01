from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PostPromptGenerator(BaseModel):
    """포스트용 프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    def create_prompt(self) -> PromptTemplate:
        """HyperCLOVA X 최적화된 간결한 프롬프트 템플릿 생성"""
        
        # SFT 파인튜닝과 동일한 형식으로 프롬프트 구성
        system_prompt = "너는 동물 유형과 감정에 맞게 문장을 자연스럽게 변환하는 전문가야."
        user_prompt = (
            f"다음 문장을 {self.emotion}한 {self.post_type} 말투로 바꿔줘.\n"
            f"Input: {self.content}\n"
            f"Output:"
        )
        
        template = (
            f"<|system|>\n{system_prompt}\n"
            f"<|user|>\n{user_prompt}\n"
            f"<|assistant|>\n"
        )

        return PromptTemplate(
            input_variables=["content"],
            template=template
        )

    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content)
