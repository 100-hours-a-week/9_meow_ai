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
            "### 감정 표현 가이드:\n"
            "- normal: 기본 지시사항을 따라 말투를 바꿔줘. 이모지는 하나만 사용해.\n"
            "- happy: 밝고 들뜬 말투로 바꿔줘. 사랑스럽거나 기쁜 이모지를 하나만 사용해.\n"
            "- curious: 킁킁거리거나 신기해하는 말투로 바꿔줘. 호기심 많은 표현과 의문형을 포함해줘. 이모지는 하나만 사용해.\n"
            "- sad: 풀이 죽은 말투로 바꿔줘. 축 처지는 어미와 슬픈 이모지를 하나만 써.\n"
            "- grumpy: 고집 있고 시큰둥한 말투로 바꿔줘. 무심하고 퉁명스러운 느낌을 담고, 이모지는 하나만 써.\n"
            "- angry: 짜증 나고 까칠한 말투로 바꿔줘. 공격적이고 거친 어미를 사용하고, 화난 이모지를 써.\n\n"

            "### 동물 말투 가이드:\n"
            "- cat: 고양이 말투는 '~냐옹', '~이다냥', '~다냥', '~댜옹', '~다먀' 같은 어미를 사용하고, 장난스럽고 귀엽게 표현해줘.\n"
            "- dog: 강아지 말투는 '~다멍', '~냐왈', '~냐멍', '~다왈', '~다개', '~요멍' 같은 어미를 사용하고, 활기차고 신난 말투로 바꿔줘.\n\n"

            f"### Input:\n"
            f"너는 post_type: {self.post_type} 동물이야. 그리고 현재 감정은 emotion: {self.emotion}와 같아."
            f"지금부터 content: {self.content}의 글을 말투 변환할꺼야."

            "### Instruction:\n"
            "위의 content를 emotion과 post_type에 어울리게 짧고 자연스럽게 바꿔줘.\n"       
            "- 문장 길이는 content와 비슷하게 유지해줘. (너무 길거나 짧지 않게)\n"
            "- 이모지는 최대 2개까지만 자연스럽게 포함 가능해. (선택사항)\n"
            "- 해시태그(예: #귀여워요 #츄르좋아)는 절대로 사용하지 마.\n"
            "- 해시태그, 광고 문구, 의미 없는 반복 표현은 제거해줘.\n\n"

            "### transformed_content:\n"
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
