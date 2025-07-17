from pydantic import BaseModel, Field
from langchain.prompts import PromptTemplate
from typing import ClassVar
import re

class CommentPromptGenerator(BaseModel):
    """커멘트용 프롬프트 생성기 클래스"""
    emotion: str = Field(..., description="감정 상태")
    post_type: str = Field(..., description="동물 타입")
    content: str = Field(..., description="변환할 원본 텍스트")
    POST_TYPE_KR: ClassVar[dict] = {"cat": "고양이", "dog": "강아지"}
    EMOTION_KR: ClassVar[dict] = {
        "normal": "평범한", "happy": "기쁜", "sad": "슬픈",
        "angry": "화난", "grumpy": "까칠한", "curious": "호기심 많은"
    }
    @staticmethod
    def preprocess(text: str) -> str:
        if not isinstance(text, str):
            return text

        # 1. 줄바꿈/탭 → 공백, 연속 공백 정리
        text = re.sub(r'[\r\n\t]', ' ', text)
        text = re.sub(r'\s+', ' ', text)

        # 2. 깨진 URL 감지 및 합치기 (마침표 보존, 공백만 제거)
        def fix_url(m):
            url = m.group(1)
            # URL 내부 공백만 제거, 마침표 등은 보존
            url_clean = re.sub(r'\s+', '', url)
            return f'[{url_clean}]'
        # URL: https로 시작, 영어/숫자/특수문자(-._~:/?#[]@!$&'()*+,;=)만 포함, 한글/공백/이모지에서 종료
        url_broken_pattern = re.compile(
            r'(https?://[A-Za-z0-9\-\._~:/\?#\[\]@!\$&\'\(\)\*\+,;=%]+)'
        )
        text = url_broken_pattern.sub(fix_url, text)

        # 3. 구두점 앞에 붙은 공백 제거
        text = re.sub(r'\s+([?.!])', r'\1', text)

        # 4. 마침표(.) 뒤에 '한글'이 나오면 공백 추가 (영어나 숫자는 영향 X)
        text = re.sub(r'(\.)([가-힣])', r'\1 \2', text)

        # 5. 기타 전처리
        text = re.sub(r'^[\s.,?!·~…]+|[\s.,?!·~…]+$', '', text)
        text = re.sub(r'\s+', ' ', text).strip()

        return text
    
    def create_prompt(self) -> PromptTemplate:
        """HyperCLOVA X 최적화된 간결한 프롬프트 템플릿 생성"""

        # emotion이 normal이 아닐 경우 에러 발생
        if self.emotion != "normal":
            raise ValueError("CommentPromptGenerator는 emotion='normal'만 허용합니다.")

        # SFT 파인튜닝과 동일한 형식으로 프롬프트 구성
        post_type_kr = self.POST_TYPE_KR.get(self.post_type, self.post_type)
        emotion_kr = self.EMOTION_KR.get(self.emotion, self.emotion)
        content_preprocess =self.preprocess(self.content)
        system_prompt = "너는 동물 유형과 감정에 맞게 문장을 자연스럽게 변환하는 전문가야."
        user_prompt = (
            f"다음 문장을 {emotion_kr}한 {post_type_kr} 말투로 바꿔줘.\n"
            f"Input: {content_preprocess}\n"
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
