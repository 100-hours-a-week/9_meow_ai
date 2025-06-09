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

    # 동물 스타일 가이드 (고정 상수)
    animal_style_guide: ClassVar[Dict[str, str]] = {
        "cat": "고양이 말투는 '~냐옹', '~이다냥', '~다냥', '~댜옹', '~다먀' 같은 어미를 사용하고, 장난스럽고 귀엽게 표현해줘.",
        "dog": "강아지 말투는 '~다멍', '~냐왈', '~냐멍', '~다왈', '~다개', '~요멍' 같은 어미를 사용하고, 활기차고 신난 말투로 바꿔줘."
    }

    # 감정 스타일 가이드 (고정 상수)
    emotion_style_guide: ClassVar[Dict[str, str]] = {
        "normal": "기본 지시사항을 따라 말투를 바꿔줘. 이모지는 하나만 사용해.",
        "happy": "밝고 들뜬 말투로 바꿔줘. 사랑스럽거나 기쁜 이모지를 하나만 사용해.",
        "curious": "킁킁거리거나 신기해하는 말투로 바꿔줘. 호기심 많은 표현과 의문형을 포함해줘. 이모지는 하나만 사용해.",
        "sad": "풀이 죽은 말투로 바꿔줘. 축 처지는 어미와 슬픈 이모지를 하나만 써.",
        "grumpy": "고집 있고 시큰둥한 말투로 바꿔줘. 무심하고 퉁명스러운 느낌을 담고, 이모지는 하나만 써.",
        "angry": "짜증 나고 까칠한 말투로 바꿔줘. 공격적이고 거친 어미를 사용하고, 화난 이모지를 써.",
    }

    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""

        # 스타일 지침 조합
        animal_guide = self.animal_style_guide.get(self.post_type, "")
        emotion_guide = self.emotion_style_guide.get(self.emotion, "")
        style_instruction = f"{animal_guide} {emotion_guide}".strip()

        # CLOVAX 형식 프롬프트
        template = (
            "### core guidelines.\n"
            "1. Transform speech without losing the meaning of the original text.\n"
            "2. Use only one emoticon per sentence.\n"
            "3. Make a clear distinction between cat and dog responses.\n"
            f"4. Style guidance: {style_instruction}\n\n"
            f"### content:\n{{content}}\n"
            f"### emotion:\n{{emotion}}\n"
            f"### post_type:\n{{post_type}}\n\n"
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
