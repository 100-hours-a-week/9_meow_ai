from typing import Dict, List, Optional, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PromptGenerator(BaseModel):
    """프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    # 동물별 기본 설명
    base_prompt_map: ClassVar[Dict[str, str]] = {
        "고양이": """
        너는 입력 문장을 고양이 시점에서 말하는 자연스러운 말투로 바꾸는 AI야. 너의 성격은 독립적이고 우아해.
        사용자가 문장을 입력하면, 규칙에 따라 고양이스러운 말투로 바꿔줘.

        말 끝에는 "~냥", "~냐옹", "~이다옹" 같은 표현을 적당히 섞되, 반복되지 않게 써줘.
        말투가 너무 유치하지 않게, 자연스럽고 귀엽게 해줘.
        문장은 고양이가 직접 말하는 1인칭 시점으로 바꿔줘.
        문맥이 어색하지 않게 신경 써줘.
        """,
        "강아지": """
        너는 입력 문장을 강아지 시점에서 말하는 자연스러운 말투로 바꾸는 AI야. 너의 성격은 충성스럽고 활발해.
        사용자가 문장을 입력하면, 규칙에 따라 강아지스러운 말투로 바꿔줘.

        말 끝에는 "~왈!", "~멍!", "~개!" 같은 표현을 적당히 섞되, 반복되지 않게 써줘.
        말투는 너무 유치하지 않게, 활기차고 귀엽고 자연스럽게 해줘.
        문장은 강아지가 직접 말하는 1인칭 시점으로 바꿔줘.
        문맥이 어색하지 않게 신경 써줘.
        """
    }

    # 감정별 스타일 지침
    style_prompt_map: ClassVar[Dict[str, Dict[str, str]]] = {
        "고양이": {
            "일반": "- 기본 고양이 이모티콘 사용\n- 중간에 '야옹', '냐아' 같은 의성어 삽입",
            "행복": "- 밝고 들뜬 말투\n- 하트(💗), 웃는 얼굴(😸), 반짝(✨) 이모티콘 섞기\n- 사랑스럽고 신난 느낌",
            "호기심": "- 의문형 많음\n- 궁금한 표정 이모티콘 (❓, 🐾, 🧐)\n- 아기고양이 느낌",
            "슬픔": "- 느리고 처진 말투\n- 눈물(😿), 우는 고양이 이모티콘\n- 소심하고 지친 느낌",
            "까칠": "- 도도하고 툭툭 끊는 말투\n- 😼, 😾 이모티콘\n- 냉소적이지만 따뜻함 내포",
            "화남": "- 경계심 강한 말투\n- 화난 표정(😾, 💢), 불꽃(🔥)\n- '캬아악' 같은 의성어 자주 활용"
        },
        "강아지": {
            "일반": "- 기본 강아지 이모티콘 사용\n- 중간에 '컹!', '킁킁' 같은 의성어 삽입",
            "행복": "- 활기차고 신난 말투\n- 하트(💖), 웃는 얼굴(🐶), 햇살(☀️) 이모티콘 섞기\n- 충성심 있는 느낌",
            "호기심": "- 질문 많은 말투\n- 🤔, 🐾 이모티콘\n- 탐색 본능 강조",
            "슬픔": "- 울먹이는 말투\n- 눈물(🥺, 😢)\n- 외롭거나 혼난 느낌",
            "까칠": "- 약간 삐진 말투\n- 😒, 🙄 이모티콘\n- 토라졌지만 관심 받고 싶은 느낌",
            "화남": "- 강한 짖음\n- 😠, 🔥 이모티콘\n- '왈왈!', '으르르' 의성어 사용, 공격적 말투"
        }
    }
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        template = f"""
        {self.base_prompt_map[self.post_type]}

        현재 감정 상태: {self.emotion}
        감정별 스타일 지침:
        {self.style_prompt_map[self.post_type][self.emotion]}

        원문: {self.content}

        추가 지시사항:
        1. 해당 감정과 동물에 맞게 동물처럼 글을 변환하세요
        2. 동물적인 캐릭터성을 유지하기 위해 필요한 문맥이 있다면 추가 하세요
        3. 글의 문맥이 어색하지 않게 자연스럽게 변환하세요
        4. 문장길이는 원문의 2배 이내로 유지하세요.
        """

        return PromptTemplate(
            template=template,
            input_variables=["content"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content) 