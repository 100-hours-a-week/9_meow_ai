import re

def cat_converter(text):
    """텍스트를 냥체로 변환하는 함수"""
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
    
    # 2. "안녕" → "안냥" 변환
    result = re.sub(r'안녕', '안냥', result)
    
    # 3. "하이" → "냥하" 변환 (새로 추가)
    result = re.sub(r'하이', '냥하', result)

    # 4. 새로운 변환 규칙들 추가
    # '-나요' → '냥' 변환
    result = re.sub(r'([가-힣]+)나요(?=[!?\s.,]|$)', r'\1냥', result)
    
    # '-가요' → '가냥' 변환  
    result = re.sub(r'([가-힣]+)가요(?=[!?\s.,]|$)', r'\1가냥', result)

    # '헐' → '먀아' 변환
    result = re.sub(r'(?<![가-힣])헐(?![가-힣])', '먀아', result)
    
    # '드립니다' → '드립니다냥' 변환
    result = re.sub(r'([가-힣]+)드립니다(?=[!?\s.,]|$)', r'\1드립니다냥', result)
    
    # 강조 부사 변환들
    result = re.sub(r'(\s)완전([\s가-힣])', r'\1냥전\2', result)  # 완전 → 냥전
    result = re.sub(r'(\s)진짜([\s가-힣])', r'\1냥짜\2', result)  # 진짜 → 냥짜  
    result = re.sub(r'(\s)정말([\s가-힣])', r'\1냥말\2', result)  # 정말 → 냥말
    result = re.sub(r'(\s)엄청([\s가-힣])', r'\1냥청\2', result)  # 엄청 → 냥청
    result = re.sub(r'(\s)되게([\s가-힣])', r'\1냥게\2', result)  # 되게 → 냥게
    result = re.sub(r'(\s)너무([\s가-힣])', r'\1냥무\2', result)  # 너무 → 냥무
    result = re.sub(r'(\s)매우([\s가-힣])', r'\1냥우\2', result)  # 매우 → 냥우
    result = re.sub(r'(\s)많이([\s가-힣])', r'\1냥이\2', result)  # 많이 → 냥이
    result = re.sub(r'(\s)조금([\s가-힣])', r'\1냥금\2', result)  # 조금 → 냥금
    result = re.sub(r'(\s)좀([\s가-힣])', r'\1냥\2', result)      # 좀 → 냥
    
    # '좋아' → '냥좋아' 변환
    result = re.sub(r'(?<![가-힣])좋아(?=[!?\s.,]|$)', '냥좋아', result)
    result = re.sub(r'(?<![가-힣])좋아요(?=[!?\s.,]|$)', '냥좋아요', result)
    
    # '졸려' 관련 변환
    result = re.sub(r'(?<![가-힣])졸려(?=[!?\s.,]|$)', '냥졸려', result)
    result = re.sub(r'(?<![가-힣])졸려요(?=[!?\s.,]|$)', '냥졸려요', result)
    
    # '대박' → '냥대박' 변환
    result = re.sub(r'(?<![가-힣])대박(?=[!?\s.,]|$)', '냥대박', result)
    
    # '~싶어' → '~싶냥' 변환
    result = re.sub(r'([가-힣]+)싶어(?=[!?\s.,]|$)', r'\1싶냥', result)
    result = re.sub(r'([가-힣]+)싶어요(?=[!?\s.,]|$)', r'\1싶냥요', result)
    
    # 'ㄱㄱ' → '고고냥' 변환
    result = re.sub(r'ㄱㄱ', '고고냥', result)
    
    # '하,' 또는 '하.' → '냐아,' 또는 '냐아.' 변환 (쉼표/마침표가 붙은 경우만)
    result = re.sub(r'(?<![가-힣])하([,.])', r'냐아\1', result)
    
    # '-지죠' → '-지냐옹' 변환
    result = re.sub(r'([가-힣]+)지죠(?=[!?\s.,]|$)', r'\1지냐옹', result)
    
    # '-자나' → '자냐아' 변환
    result = re.sub(r'([가-힣]+)자나(?=[!?\s.,]|$)', r'\1자냐아', result)
    
    # '-임' → '-이다냥' 변환
    result = re.sub(r'([가-힣]+)임(?=[!?\s.,]|$)', r'\1이다냥', result)
    
    # '-잖아' → '-잖냐옹' 변환
    result = re.sub(r'([가-힣]+)잖아(?=[!?\s.,~]|$)', r'\1잖냐옹', result)
    
    # 과거형 어미 변환들
    # '-겁니다' → '-거다냥' 변환 (긴 패턴 먼저)
    result = re.sub(r'([가-힣]+)겁니다(?=[!?\s.,]|$)', r'\1거다냥', result)
    
    # '-군' → '-구냐아' 변환
    result = re.sub(r'([가-힣]+)군(?=[!?\s.,]|$)', r'\1구냐아', result)
    
    # 특별 형용사 변환
    # '귀엽다' → '귀엽다냐하' 변환
    result = re.sub(r'귀엽다(?=[!?\s.,]|$)', '귀엽다냐하', result)

    # 5. 대답 변환: "응" → "냥", "네" → "냥", "예" → "녜" (제한적)
    result = re.sub(r'^응(?=[!?\s.,]|$)', '냥', result)
    result = re.sub(r'(\s)응(?=[!?\s.,]|$)', r'\1냥', result)
    # "네"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^네([!?.,])', r'냥\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^네(?=\s*$)', '냥', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)네([!?.,])', r'\1냥\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)네(?=\s*$)', r'\1냥', result)  # 중간에 단독으로 끝나는 경우
    # "예"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^예([!?.,])', r'녜\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^예(?=\s*$)', '녜', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)예([!?.,])', r'\1녜\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)예(?=\s*$)', r'\1녜', result)  # 중간에 단독으로 끝나는 경우
    
    # 6. 감탄사 변환: 문장 맨 앞의 감탄사 변환 (문장 끝 처리 전에 실행)
    result = re.sub(r'^와!', '냐아!', result)
    result = re.sub(r'^오!', '냐아!', result)
    result = re.sub(r'^아!', '냐아!', result)
    
    # 문두 단독 감탄사도 변환 (문장부호 없이) - 앞뒤에 한글이 없는 경우만
    result = re.sub(r'^와(?=\s)', '냐아', result)  # 공백 앞의 "와"
    result = re.sub(r'^오(?=\s)', '냐아', result)  # 공백 앞의 "오"  
    result = re.sub(r'^아(?=\s)', '냐아', result)  # 공백 앞의 "아"
    result = re.sub(r'^(오|아|와)$', '냐아', result)
    
    # 7. 감탄사 변환 (앗, 앙, 으악, 아악) - 위치에 관계없이 모두 변환
    result = re.sub(r'(?<![가-힣])앙(?![가-힣])', '냐앙', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])앗(?![가-힣])', '냐앗', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])으악(?![가-힣])', '냐악', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])아악(?![가-힣])', '냐악', result)  # 앞뒤에 한글이 없는 경우
    
    # 8. 자음 조합 변환 (긴 패턴부터 먼저 처리)
    result = re.sub(r'ㅎㅇㅌ', '냥이팅', result)  # ㅎㅇ보다 먼저 처리
    result = re.sub(r'ㅎㅇ', '하이다냥~', result)  # ㅎㅇ → 하이다냥~ 변환
    result = re.sub(r'ㅇㅁ', '어머냥', result)
    result = re.sub(r'ㅁㅇ', '모냥', result)
    result = re.sub(r'ㄱㅊ', '괜찮냥', result)  # ㄱㅊ → 괜찮냥
    # ㄱㅇㅇ를 임시로 보호
    result = re.sub(r'ㄱㅇㅇ', 'TEMP_GYY', result)
    # ㅇㅇ 변환
    result = re.sub(r'ㅇㅇ', '웅냥', result)
    # ㄱㅇㅇ 복원
    result = re.sub(r'TEMP_GYY', 'ㄱㅇㅇ', result)
    
    result = re.sub(r'ㅇㄸ', '어떠냥', result)
    result = re.sub(r'(?<![가-힣])아하(?![가-힣])', '냐하', result)  # 앞뒤에 한글이 없는 독립된 "아하"만
    
    
    # 새로운 자음/모음 변환 규칙들
    result = re.sub(r'ㅋㅋ+', r'\g<0>냥하하', result)  # ㅋㅋ → ㅋㅋ냥하하 (뒤에 추가)
    result = re.sub(r'ㅎㅎ+', r'\g<0>먀하하', result)  # ㅎㅎ → ㅎㅎ먀하하
    result = re.sub(r'ㅜ+', '냐아..', result)  # ㅜ → 냐아..

    # 9. 특별 단어/어절 처리
    # "개웃" → "냥웃" (개웃겨, 개웃기다, 개웃김 등)
    result = re.sub(r'개웃', '냥웃', result)

        # 특정 이름 변환
    result = re.sub(r'(?<![가-힣])조이(?![가-힣])', '조이냥이', result)
    result = re.sub(r'(?<![가-힣])두식이(?![가-힣])', '두식냥이', result)
    result = re.sub(r'(?<![가-힣])임절미(?![가-힣])', '임절멍이', result)
    result = re.sub(r'(?<![가-힣])텐시(?![가-힣])', '텐시멍이', result)
    
    # 공백 뒤 강조 표현: "개이쁘", "개귀엽", "개귀여" → "냥이쁘", "냥귀엽", "냥귀여" (강조 용법만)
    result = re.sub(r'(\s)개(이쁘|귀엽|귀여)', r'\1냥\2', result)  # 공백 뒤에만
    
    # "존" 강조 표현 변환 (공백 뒤에만)
    result = re.sub(r'(\s)존(잼|맛|맛탱|예|귀|좋)', r'\1냥\2', result)  # 공백 뒤에만
    
    # 뒤에 한글이 오지 않는 경우에만 냥 붙이기
    result = re.sub(r'(맞아|마자|마좌|마쟈)(?![가-힣])', r'\1냥', result)

    # 다옹 냐옹 
    result = re.sub(r'([가-힣])(다|나|냐)\b', r'\1\2옹', result)
    result = re.sub(r'([가-힣])요\b', r'\1야옹', result)
    # 10. 문장 끝에 "냥" 추가 (한국어가 포함된 경우만)
    if re.search(r'[가-힣]', result):
        # 문장 끝 처리 - 문장부호로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!냥)(\s*[.!?~\\,;]+)', r'\1냥\2', result)  # 문장부호로 끝 (공백 포함)
        
        # 이모티콘으로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!냥)(\s*\^\^\s*$)', r'\1냥\2', result)  # ^^ 이모티콘
        result = re.sub(r'([가-힣])(?<!냥)(\s*:\)\s*$)', r'\1냥\2', result)  # :) 이모티콘
        result = re.sub(r'([가-힣])(?<!냥)(\s*[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+\s*$)', r'\1냥\2', result)  # 유니코드 이모티콘
        
        # 자음/모음(ㅋㅋ, ㅎㅎ, ㅇㅋ 등) 앞의 한글에 "냥" 추가
        result = re.sub(r'([가-힣])(?<!냥)(\s*[ㄱ-ㅎㅏ-ㅣㅋㅎㅇㅋ]+)', r'\1냥\2', result)  # 자음/모음 앞
        
        # 줄바꿈 처리
        result = re.sub(r'([가-힣])(?<!냥)(\s*\r?\n)', r'\1냥\2', result)  # 줄바꿈으로 끝
        
        # 문장부호 없이 끝나는 경우
        result = re.sub(r'([가-힣])(?<!냥)(\s*$)', r'\1냥\2', result)  # 그냥 끝나는 경우
    
    # 11. 감탄사 변환: "와!" → "냐아!", "오!" → "냐아!" (문장 끝 처리 후에 실행)
    result = re.sub(r'^와!', '냐아!', result)
    result = re.sub(r'(\s)와!', r'\1냐아!', result)
    result = re.sub(r'^오!', '냐아!', result)
    result = re.sub(r'(\s)오!', r'\1냐아!', result)
    
    # 12. 불필요한 "냥" 제거 (특별 변환 후 붙은 냥 정리)
    # 단일 패턴 뒤의 냥 제거
    result = re.sub(r'(냐앙|냐앗|냐악|어떠냥|냐하|어머냥|모냥|괜찮냥|냥이팅|녜|냐아|냥잼|냥맛|냥맛탱|냥예|냥귀|냥좋|옹|먀하하|냐하하|냥이|구냐아|먀아|냥웃겨|냥웃기다|냥웃김|냥웃곀)냥', r'\1', result)
    
    # 연속 패턴의 마지막에만 냥 남기기 (예: 냥하냥하냥 → 냥하냥하)
    result = re.sub(r'(냥하)+냥(?![냥하])', lambda m: m.group(0)[:-1], result)  # 냥하 연속 후 마지막 냥만 제거
    result = re.sub(r'(웅냥)+냥(?![웅냥])', lambda m: m.group(0)[:-1], result)  # 웅냥 연속도 같은 방식
    
    # 13. "아아" 복원
    result = re.sub(r'TEMP_AA', '아아', result)
    
    # 14. 작은따옴표 내용 복원 (맨 마지막에)
    for placeholder, original in quoted_parts.items():
        result = result.replace(placeholder, original)
    
    return result

