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
            1. 사용자가 입력한 댓글을 고양이가 1인칭으로 말하는 말로 바꿔줘.
            2. 말투는 반드시 '~냥', '~냐옹', '~이다옹', '~옹', '~다옹' 등의 어미로 끝나야 한다.
            3. 'ㅋㅋㅋ'는 '냐하하!'로, 'ㅎㅎㅎ'는 '먀하하!'로 바꾸되, 각 표현은 한 번만 사용하라.
            7. 반드시 한국어로만 작성한다.
            """,

        "dog": """
            [역할] 너는 강아지의 말투와 문맥으로 댓글을 바꾸는 변환기다.
            
            [규칙]
            1. 사용자가 입력한 댓글을 강아지가 1인칭으로 말하는 것처럼 바꿔줘. 
            2. 말투는 반드시 '~다멍', '~냐왈', '~다컹', '~냐멍', '~다왈', '~다개' 등의 어미로 끝나야 한다.
            4. 반드시 한국어로만 작성한다.

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