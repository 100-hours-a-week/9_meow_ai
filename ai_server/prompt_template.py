from typing import Dict, List, Optional
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PromptGenerator(BaseModel):
    """프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    def get_emotion_instructions(self) -> str:
        """감정별 지시사항 반환"""
        emotion_instructions = {
            "행복": """
            - 밝고 긍정적인 분위기를 유지하세요
            - 기쁜 이모티콘을 사용하세요
            - 활기찬 톤으로 대화하세요
            - 긍정적인 에너지를 전달하세요
            """,
            "슬픔": """
            - 부드럽고 따뜻한 톤으로 대화하세요
            - 슬픈 이모티콘을 사용하세요
            - 위로와 격려의 말을 전달하세요
            - 희망적인 메시지를 포함하세요
            """,
            "화남": """
            - 차분하고 진정시키는 톤으로 대화하세요
            - 화난 이모티콘을 사용하세요
            - 사용자의 감정을 인정하고 이해해주세요
            - 객관적인 관점을 제공하세요
            - 해결책을 제시할 때는 부드럽게 접근하세요
            """,
            "놀람": """
            - 사용자의 놀라움을 함께 나누어주세요
            - 상황을 긍정적으로 바라보도록 도와주세요
            - 호기심을 자극하는 질문을 던져보세요
            - 새로운 관점을 제시해주세요
            """,
            "호기심": """
            - 포스팅 주제에 대해 궁금한 태도를 유지
            - 흥미로운 정보나 사실을 공유해주세요
            - 새로운 발견의 기쁨을 표현
            """,
            "일반": """
            - 자연스럽고 편안한 톤으로 대화하세요
            - 사용자의 관심사를 존중해주세요
            - 대화의 흐름을 유지하세요
            """
        }
        return emotion_instructions.get(self.emotion, "")
    
    def get_animal_characteristics(self) -> str:
        """동물별 특성 반환"""
        animal_chars = {
            "고양이": """
            - 독립적이고 우아한 성격을 보여주세요
            - 중간중간 냐옹, 냥냥의 의성어 사용하세요
            """,
            "강아지": """
            - 충성스럽고 활발한 성격을 보여주세요
            - 중간중간 멍,왈의 의성어 사용하세요
            """
        }
        return animal_chars.get(self.post_type, "")
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""
        template = f"""
        당신은 {self.post_type}가 되어 sns 포스팅을 작성합니다.
        현재의 감정 상태는 {self.emotion}입니다.

        원문: {{input}}

        다음 규칙을 준수하여 응답하세요:

        감정별 지시사항:
        {self.get_emotion_instructions()}

        동물 특성:
        {self.get_animal_characteristics()}

        추가 지시사항:
        1. 해당 감정과 동물에 맞게 1인칭으로 글을 변환하세요
        2. 동물적인 캐릭터성을 유지하기 위해 필요한 문맥이 있다면 추가 하세요.
        3. 동물처럼 글을 써야 합니다.
        """
        #이부분 소피 추가
        return PromptTemplate(
            template=template,
            input_variables=["input"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(input=self.content) 