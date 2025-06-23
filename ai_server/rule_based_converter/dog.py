import re

def dog_converter(text):
    """텍스트를 멍체로 변환하는 함수"""
    if not text or not isinstance(text, str):
        return text
    
    result = text
    
    # 0. 작은따옴표 안의 내용 보호
    quoted_parts = {}
    quote_pattern = r"'([^']*?)'"
    
    def replace_quoted(match):
        placeholder = f"TEMP_QUOTE_{len(quoted_parts)}"
        quoted_parts[placeholder] = match.group(0)
        return placeholder
    
    result = re.sub(quote_pattern, replace_quoted, result)
    
    # 1. "아아" 보호 (아이스아메리카노, 의성어)
    result = re.sub(r'아아', 'TEMP_AA', result)
    
    # 2. "안녕" → "멍하" 변환
    result = re.sub(r'안녕(?![하히])', '멍하', result)
    
    # 3. "하이" → "멍하" 변환 (새로 추가)
    result = re.sub(r'하이', '멍하', result)
    
    
    # 5. 대답 변환: "응" → "왈", "네" → "왈", "예" → "왈" (제한적)
    result = re.sub(r'^응(?=[!?\s.,]|$)', '왈', result)
    result = re.sub(r'(\s)응(?=[!?\s.,]|$)', r'\1왈', result)
    # "네"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^네([!?.,])', r'왈\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^네(?=\s*$)', '왈', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)네([!?.,])', r'\1왈\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)네(?=\s*$)', r'\1왈', result)  # 중간에 단독으로 끝나는 경우
    # "예"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^예([!?.,])', r'왈\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^예(?=\s*$)', '왈', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)예([!?.,])', r'\1왈\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)예(?=\s*$)', r'\1왈', result)  # 중간에 단독으로 끝나는 경우
    
    # 6. 감탄사 변환: 문장 맨 앞의 감탄사 변환 (문장 끝 처리 전에 실행)
    result = re.sub(r'^와!', '왕왕!', result)
    result = re.sub(r'^오!', '왕왕!', result)
    result = re.sub(r'^아!', '왕왕!', result)
    
    # 문두 단독 감탄사도 변환 (문장부호 없이) - 앞뒤에 한글이 없는 경우만
    result = re.sub(r'^와(?=\s)', '왕왕', result)  # 공백 앞의 "와"
    result = re.sub(r'^오(?=\s)', '왕왕', result)  # 공백 앞의 "오"  
    result = re.sub(r'^아(?=\s)', '왕왕', result)  # 공백 앞의 "아"
    result = re.sub(r'^(오|아|와)$', '왕왕', result)  # 단독으로 "오", "아", "와"가 있는 경우
    
    # 7. 감탄사 변환 (앗, 앙, 으악, 아악) - 위치에 관계없이 모두 변환
    result = re.sub(r'(?<![가-힣])앙(?![가-힣])', '컹', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])앗(?![가-힣])', '컹', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])으악(?![가-힣])', '으르렁', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])아악(?![가-힣])', '으르렁', result)  # 앞뒤에 한글이 없는 경우
    
    # 8. 자음 조합 변환 (긴 패턴부터 먼저 처리)
    result = re.sub(r'ㅎㅇㅌ', '멍이팅', result)  # ㅎㅇ보다 먼저 처리
    result = re.sub(r'ㅎㅇ', '멍하', result)
    result = re.sub(r'ㅇㅁ', '어멍', result)
    result = re.sub(r'ㅁㅇ', '모냐멍', result)
    result = re.sub(r'ㄱㅊ', '괜찮컹', result)  # ㄱㅊ → 괜찮컹
    # ㄱㅇㅇ를 임시로 보호
    result = re.sub(r'ㄱㅇㅇ', 'TEMP_GYY', result)
    # ㅇㅇ 변환
    result = re.sub(r'ㅇㅇ', '웅왈', result)
    # ㄱㅇㅇ 복원
    result = re.sub(r'TEMP_GYY', 'ㄱㅇㅇ', result)
    
    result = re.sub(r'ㅇㄸ', '어뗘컹', result)
    result = re.sub(r'(?<![가-힣])아하(?![가-힣])', '아하컹', result)  # 앞뒤에 한글이 없는 독립된 "아하"만
    
    # 9. 특별 단어/어절 처리
    # "개웃" → "댕웃" (개웃겨, 개웃기다, 개웃김 등)
    result = re.sub(r'개웃', '댕웃', result)
    
    # 공백 뒤 강조 표현: "개이쁘", "개귀엽", "개귀여" → "댕이쁘", "댕귀엽", "댕귀여" (강조 용법만)
    result = re.sub(r'(\s)개(이쁘|귀엽|귀여)', r'\1댕\2', result)  # 공백 뒤에만
    
    # "존" 강조 표현 변환 (공백 뒤에만)
    result = re.sub(r'(\s)존(잼|맛|맛탱|예|귀|좋)', r'\1댕\2', result)  # 공백 뒤에만
    
    # 뒤에 한글이 오지 않는 경우에만 멍 붙이기
    result = re.sub(r'(맞아|마자|마좌|마쟈)(?![가-힣])', r'\1컹', result)
    
    # 10. 문장 끝에 "멍" 추가 (한국어가 포함된 경우만)
    if re.search(r'[가-힣]', result):
        # 문장 끝 처리 - 문장부호로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!멍)(\s*[.!?~\\,;]+)', r'\1멍\2', result)  # 문장부호로 끝 (공백 포함)
        
        # 이모티콘으로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!멍)(\s*\^\^\s*$)', r'\1멍\2', result)  # ^^ 이모티콘
        result = re.sub(r'([가-힣])(?<!멍)(\s*:\)\s*$)', r'\1멍\2', result)  # :) 이모티콘
        result = re.sub(r'([가-힣])(?<!멍)(\s*[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+\s*$)', r'\1멍\2', result)  # 유니코드 이모티콘
        
        # 자음/모음(ㅋㅋ, ㅎㅎ, ㅇㅋ 등) 앞의 한글에 "멍" 추가
        result = re.sub(r'([가-힣])(?<!멍)(\s*[ㄱ-ㅎㅏ-ㅣㅋㅎㅇㅋ]+)', r'\1멍\2', result)  # 자음/모음 앞
        
        # 줄바꿈 처리
        result = re.sub(r'([가-힣])(?<!멍)(\s*\r?\n)', r'\1멍\2', result)  # 줄바꿈으로 끝
        
        # 문장부호 없이 끝나는 경우
        result = re.sub(r'([가-힣])(?<!멍)(\s*$)', r'\1멍\2', result)  # 그냥 끝나는 경우
    
    # 11. 감탄사 변환: "와!" → "왕왕!", "오!" → "왕왕!" (문장 끝 처리 후에 실행)
    result = re.sub(r'^와!', '왕왕!', result)
    result = re.sub(r'(\s)와!', r'\1왕왕!', result)
    result = re.sub(r'^오!', '왕왕!', result)
    result = re.sub(r'(\s)오!', r'\1왕왕!', result)
    
    # 12. 불필요한 "멍" 제거 (특별 변환 후 붙은 멍 정리)
    # 단일 패턴 뒤의 멍 제거
    result = re.sub(r'(컹|미스코리냥|#미스코리냥|왈왈|왈|왕왕|어뗘컹|냐하|어멍|모냐멍|괜찮컹|멍이팅|녜|왕왕|댕잼|댕맛|댕맛탱|댕예|댕귀|댕좋|아하컹|맞아컹|마자컹|마좌컹|마쟈컹|어뗘컹)멍', r'\1', result)
    
    # 연속 패턴의 마지막에만 멍 남기기 (예: 멍하멍하멍 → 멍하멍하)
    result = re.sub(r'(멍하)+멍(?![멍하])', lambda m: m.group(0)[:-1], result)  # 멍하 연속 후 마지막 멍만 제거
    result = re.sub(r'(웅왈)+멍(?![웅왈])', lambda m: m.group(0)[:-1], result)  # 웅멍 연속도 같은 방식
    
    # 13. "아아" 복원
    result = re.sub(r'TEMP_AA', '아아', result)
    
    # 14. 작은따옴표 내용 복원 (맨 마지막에)
    for placeholder, original in quoted_parts.items():
        result = result.replace(placeholder, original)
    
    return result




