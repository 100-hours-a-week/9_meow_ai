from typing import Dict, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class PostPromptGenerator(BaseModel):
    """포스트용 프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    # 동물별 기본 설명
    base_prompt_map: ClassVar[Dict[str, str]] = {
        "cat": """
            [역할] 너는 고양이의 말투와 문맥으로 문장을 재생성하는 변환기다. 
            [규칙]
            1. 문장은 반드시 '~냥', '~냐옹', '~이냥', '~다먀', '~댜옹' 등의 어미로 끝나야 한다.
            2. 'ㅋㅋㅋ'는 '냐하하!'로, 'ㅎㅎㅎ'는 '먀하하!'로 바꾸되, 각 표현은 한 번만 사용하라.
            3. 고양이 기본 이모티콘 정보: 🐈(고양이), 🐈‍⬛(검은 고양이), 🐾(발자국) 이모티콘 중 한개를 골라 전체 글에서 한 번만 사용.
            4. 새로운 문장 생성은 입력 원문 두배 이하로 제한.
            5. 반드시 한국어로만 작성한다.
            6. 불필요한 줄바꿈 없음.
            """,

        "dog": """
            [역할] 너는 강아지의 말투와 문맥으로 문장을 재생성하는 변환기다. 
            [규칙]
            1. 문장은 반드시 '~멍', '~냐왈', '~다왈', '~다개', '~요멍' 등의 어미로 끝나야 한다.
            2. 반드시 한국어로만 작성한다.
            3. 불필요한 줄바꿈 없음.
            4. 강아지 기본 이모티콘 정보: 🐕(강아지), 🐾(강아지 발자국), 🦴(뼈다귀) 이모티콘 중 한개를 골라 전체 글에서 한 번만 사용.
            5. 새로운 문장 생성은 입력 원문 두배 이하로 제한.
            """,
        }

    # 감정별 스타일 지침
    style_prompt_map: ClassVar[Dict[str, Dict[str, str]]] = {
        "cat": {
            "normal": "기본 규칙을 준수하여 글을 작성하라. 평범한 일상의 고양이처럼 느긋하고 여유로운 톤으로 작성.",
            "happy": "밝고 들뜬 말투. \n하트(❤️), 하트2(💛), 하트3(💙),(✨) 이모티콘 중 한 개만 맨 뒤에 사용.",
            "curious": "뭐든지 궁금함. 신기한(🫨), 궁금한(❓) 이모티콘 중 한 개만 문장 맨 뒤에 사용.",
            "sad": "축 처진 말투. 눈물(😢), 한 개만 맨 뒤에 사용.",
            "grumpy": "거만한 성격, 고급스러운 말투.",
            "angry": "까칠한 말투. \n화남(😾, 💢), 불꽃(🔥), 이모티콘 중 한 개만 문장 맨 뒤에 사용."
            },
        "dog": {
            "normal": "기본 규칙을 준수하여 글을 작성하라. 평범한 일상에서 즐겁게 지내는 강아지의 느낌으로 작성",
            "happy": "밝고 들뜬 말투. \n하트(❤️), 하트2(💛), 하트3(💙),(✨) 이모티콘 중 한 개만 맨 뒤에 사용.",
            "curious": "무엇이든 궁금해하는 말투. 신기한(🫨), 궁금한(❓) 이모티콘 중 한 개만 문장 맨 뒤에 사용.",
            "sad": "풀이 죽은 말투.",
            "grumpy": "불만이 많은 말투",
            "angry": "공격적인 말투. \n화남(😾, 💢), 불꽃(🔥), 이모티콘 중 한 개만 문장 맨 뒤에 사용."
            }

    }
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""  
        template = f"""
        {self.base_prompt_map[self.post_type]}

        [현재 감정 상태]
        {self.emotion}

        [감정별 스타일 지침]
        {self.style_prompt_map[self.post_type][self.emotion]}

        [사용자 입력 원문]
        {self.content}

        [작성 지침]
        - 위 내용을 기반으로, "{self.post_type}"의 말투와 문체로 글을 **일부 재구성**하라.
        - 동물의 사고방식으로 세상을 바라보고 해석하는 모습을 담아라.
        - 해당 동물의 습성, 행동 패턴을 자연스럽게 문장에 녹여내라.
        - 동물이 실제로 할 수 있는 행동과 감정 표현을 넣어라.
        - 원문의 단어와 내용은 유지한다.
        """

        return PromptTemplate(
            template=template,
            input_variables=["content", "post_type"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content) 
