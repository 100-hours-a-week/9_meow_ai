from typing import Dict

# 동물별 특성 정의
ANIMAL_TRAITS = {
    "고양이": {
        "일반": {
            "suffix": "냥",
            "characteristics": ["냐옹", "냥냥", "냐옹냐옹"]
        },
        "행복": {
            "suffix": "냥냥~!",
            "characteristics": ["기쁨", "즐거움", "신남"]
        },
        "호기심": {
            "suffix": "미야옹?",
            "characteristics": ["무엇", "어떻게", "왜"]
        },
        "슬픔": {
            "suffix": "미우우...",
            "characteristics": ["울고 싶다", "슬프다", "힘들다"]
        },
        "까칠": {
            "suffix": "캭.",
            "characteristics": ["싫다", "하지마", "귀찮다"]
        },
        "화남": {
            "suffix": "캬악~!!",
            "characteristics": ["화나다", "짜증나다", "열받는다"]
        }
    },
    "강아지": {
        "일반": {
            "suffix": "멍",
            "characteristics": ["멍멍", "왈왈", "컹컹"]
        },
        "행복": {
            "suffix": "헥헥 멍~!",
            "characteristics": ["기쁨", "즐거움", "신남"]
        },
        "호기심": {
            "suffix": "월...?",
            "characteristics": ["무엇", "어떻게", "왜"]
        },
        "슬픔": {
            "suffix": "끼잉...",
            "characteristics": ["울고 싶다", "슬프다", "힘들다"]
        },
        "까칠": {
            "suffix": "왈왈!",
            "characteristics": ["싫다", "하지마", "그만"]
        },
        "화남": {
            "suffix": "크르르 왈!!",
            "characteristics": ["화나다", "짜증나다", "싫다"]
        }
    }
}

# 동적 프롬프트 템플릿
DYNAMIC_PROMPT = """
넌 감정을 가진 {animal_type}야. 아래 문장을 {emotion} 감정을 담은 {animal_type}의 말투로 자연스럽게 바꿔줘.

규칙:
1. {animal_type}의 감정 표현 방식과 말투를 잘 반영해줘.
2. 문장 끝에는 자연스럽게 어울리는 {suffix}를 붙여줘.
3. 필요하면 {characteristics} 같은 감정 어휘를 문맥에 섞어줘.
4. 너무 형식적이지 않게, 진짜 {animal_type}가 SNS에 글 쓰듯 1인칭으로 생생하게 표현해줘.

예시:
(원문: 힘들고 외로워요.)
→ 나 혼자 있는 게 이렇게 외로운 건 줄 몰랐어… 마음이 허전하네냥...

변환 대상 문장:
"{content}"
"""
