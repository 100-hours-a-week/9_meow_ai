from typing import Dict

PROMPT_TEMPLATES: Dict[str, str] = {
    "cat_normal": """
    다음 텍스트를 고양이 말투로 변환해주세요.
    원문: {content}
    """,
    "cat_happy": """
    다음 텍스트를 행복한 고양이 말투로 변환해주세요.
    원문: {content}
    """,
    "cat_curious": """
    다음 텍스트를 호기심 많은 고양이 말투로 변환해주세요.
    원문: {content}
    """,
    "cat_sad": """
    다음 텍스트를 슬픈 고양이 말투로 변환해주세요.
    원문: {content}
    """,
    "cat_grumpy": """
    다음 텍스트를 까칠한 고양이 말투로 변환해주세요.
    원문: {content}
    """,
    "cat_angry": """
    다음 텍스트를 화난 고양이 말투로 변환해주세요.
    원문: {content}
    """
} 