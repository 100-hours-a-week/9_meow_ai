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
        
        template = f"""### Role
당신은 {self.post_type}의 관점에서 인간의 텍스트를 변환하는 전문 작가입니다.

### System Instructions
1. 목적: 입력 텍스트를 {self.post_type}의 {self.emotion}한 말투로 자연스럽게 변환
2. 톤: {self.emotion}한 감정을 {self.post_type} 특성에 맞게 표현
3. 출력형식: 변환된 텍스트만 반환 (추가 설명 없음)
4. 길이: 원문 대비 1.5-2배 이내로 제한
5. 규칙: 자연스러운 한국어, 동물 특성 반영

### Context
<transformation_context>
동물 타입: {self.post_type.upper()}
감정 상태: {self.emotion.upper()}
변환 목표: {self.post_type}가 {self.emotion}할 때의 자연스러운 표현으로 변환
</transformation_context>

### Few-shot Examples
<example>
<original>산책하다가 예쁜 노을을 봐서 기분이 좋아졌어요</original>
<{self.post_type}_{self.emotion}>
{self._get_example()}
</{self.post_type}_{self.emotion}>
</example>

### Task
다음 텍스트를 위 조건에 맞게 변환해주세요:

<original_text>
{{content}}
</original_text>

### Output Format
변환된 텍스트만 반환하세요. 추가 설명이나 태그는 포함하지 마세요.
"""

        return PromptTemplate(
            input_variables=["content"],
            template=template
        )

    def _get_example(self) -> str:
        """모든 감정-동물 조합별 예제 반환"""
        examples = {
            ("cat", "happy"): "🐾 산책하다가 냐옹~ 예쁜 노을을 보니 냐하하! 기분이 아주 좋아졌댜옹! ✨",
            ("cat", "normal"): "🐾 냐옹~ 산책하다가 냐옹~ 예쁜 노을을 보니 기분이 아주 좋아졌댜옹! 먀하하! 정말 황홀한 풍경이였다냥!",
            ("cat", "grumpy"): "흥, 인간들이나 노을 보고 좋아하지. 🐾 발이나 핥아야겠다냥. 산책하다가 예쁜 노을을 봤다니, 고양이님인 내가 보기엔 그저 붉은 하늘일 뿐이다냥! 먀하하! 기분이 좋아졌다고? 웃기지 말라냥!",
            ("cat", "angry"): "흥, 산책하다가 예쁜 노을을 봤다냥! 그래서 기분이 좀 풀린 거 같으냐옹! 🐾🔥",
            ("cat", "curious"): "🐾 킁킁, 냥냥! 산책하다가 냐옹, 예쁜 노을을 봤다냥! 기분이 좋아진 거 같다냥? 저게 뭐냥❓",
            ("cat", "sad"): "🐾 냐옹... 산책하다가... 냐옹... 예쁜 노을을 봤다냥... 그래도 슬픈 건 여전하다먀... 😢",
            ("dog", "happy"): "킁킁! 산책하다가 멍! 예쁜 노을을 봤다 멍! 꼬리 살랑살랑 너무 기분 좋다개! 🐩✨",
            ("dog", "normal"): "킁킁! 산책하다가 멍! 예쁜 노을을 봤다 멍! 꼬리 살랑살랑, 기분이 엄청 좋아졌다 왈! 🐩",
            ("dog", "grumpy"): "흥, 산책하다가 예쁜 노을 봤다개. 잠깐 기분 좋아진 건 맞다왈. 🐩 하지만 간식 안 줘서 여전히 불만이다멍!",
            ("dog", "angry"): "산책하다가 예쁜 노을 봤다왈! 근데 왜 집에 안 가는거냐왈! 빨리 🦴 간식 내놓으라개! 😾",
            ("dog", "curious"): "킁킁, 산책하다가 멍! 예쁜 노을을 봤다 멍! 꼬리 살랑살랑, 기분이 엄청 좋아졌다 왈! 저 빨간 건 뭐다 멍? 🐩",
            ("dog", "sad"): "산책하다가 예쁜 노을 봤다 멍... 🐩 근데 집에 가야 한다 왈... 꼬리도 축 쳐진다 멍..."
        }
        
        return examples.get((self.post_type, self.emotion), "예제를 찾을 수 없습니다.")

    def get_formatted_prompt(self) -> str:
        """포맷팅된 프롬프트 반환"""
        prompt_template = self.create_prompt()
        return prompt_template.format(content=self.content)
