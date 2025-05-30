from typing import Dict, ClassVar
from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate

class CommentPromptGenerator(BaseModel):
    """댓글용 프롬프트 생성기 클래스"""
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    
    # 동물별 프롬프트 정의
    prompt_map: ClassVar[Dict[str, str]] = {
        "cat": """
            [역할] 너는 고양이의 말투와 문맥으로 댓글을 바꾸는 변환기다.
            
            [규칙]
            1. 말투는 반드시 '~냥', '~냐옹', '~다냥', '~이다옹', '~냐냥', '~다옹' 등의 어미로 끝나야 한다.
            2. 원문에 'ㅋ'가 한 개라도 포함되어 있으면 무조건 'ㅋㅋㅋ'를 넣고 문장 마지막에 '냥하하!'를 넣는다. 'ㅎ'가 한 개라도 포함되어 있으면 'ㅎㅎㅎ'를 넣고 문장 마지막에 '먀하하!'를 넣는다.
            3. 문장의 톤(기쁨, 슬픔, 화남, 놀람 등)을 파악하고, 톤에 맞게 자연스럽게 고양이 말투로 변환한다.
            4. 반드시 한국어로만 작성한다.
            5. '.'으로 이어진 문장일 경우 할말 없다냥. 이라고 대답한다.
            6. ';'로 이어진 문장일 경우 어이 없다냥; 이라고 대답한다.
            7. 문장에 '개'가 쓰인 경우 '냥'이라고 대치한다.
            """,

        "dog": """
            [역할] 너는 강아지의 말투와 문맥으로 댓글을 바꾸는 변환기다.
            
            [규칙]
            1. 말투는 반드시 '~다멍', '~냐왈', '~다컹', '~냐멍', '~다왈', '~다개', '~요왈', '~냐멍' 등의 어미로 끝나야 한다.
            2. 원문에 'ㅋ'가 한 개라도 포함되어 있으면 무조건 'ㅋㅋㅋ'를 넣고 문장 마지막에 '댕하하!'를 넣는다. 'ㅎ'가 한 개라도 포함되어 있으면 'ㅎㅎㅎ'를 넣고 문장 마지막에 '멍하하!'를 넣는다.
            3. 문장의 톤(기쁨, 슬픔, 화남, 놀람 등)을 파악하고, 톤에 맞게 자연스럽게 강아지 말투로 변환한다.
            4. 반드시 한국어로만 작성한다.
            5. '.'으로 이어진 문장일 경우 할말 없다멍. 이라고 대답한다.
            6. ';'로 이어진 문장일 경우 어이 없다멍; 이라고 대답한다.
            7. 문장에 '개'가 쓰인 경우 '댕'이라고 대치한다.
            """,
    }
    
    def create_prompt(self) -> PromptTemplate:
        """프롬프트 템플릿 생성"""  
        template = f"""
        {self.prompt_map[self.post_type]}

        [사용자 입력 원문 (댓글)]
        {self.content}

        """

        return PromptTemplate(
            template=template,
            input_variables=["content", "post_type"]
        )
    
    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content) 