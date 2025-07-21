import re

def hamster_converter(text):
    """텍스트를 햄체로 변환하는 함수"""
    if not text or not isinstance(text, str):
        return text
    
    result = text
       
    # 영어 문장 체크 (알파벳, 공백, 숫자, 기본 문장부호만 포함)
    if re.match(r'^[a-zA-Z\s\d.,!?;:\'"-]+$', result.strip()):
        # 문장 끝에 문장부호가 있는 경우 앞에 meow 추가
        if re.search(r'[.!?]$', result.strip()):
            result = re.sub(r'([.!?])$', r' squeak🐹\1', result.strip())
        else:
            # 문장부호가 없는 경우 그냥 meow 추가
            result = result.strip() + ' squeak🐹'
        return result
    
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
    
    # 2. "안녕" → "햄하" 변환
    result = re.sub(r'안녕(?![하히])', '햄하', result)
    result = re.sub(r'바이', '햄바', result)
    result = re.sub(r'빠이', '햄빠', result)    
    
    result = re.sub(r'(미야옹즈|미야옹)', r'✨\1✨', result)
    result = re.sub(r'해보\b', '해보(바보)🐈', result)
    result = re.sub(r'소피 바보\b', '소피는 너무 예쁘다❤️', result)
    result = re.sub('소피', '소피🎀', result)
    result = re.sub('sophie', 'sophie🎀', result)
    result = re.sub(r'(테리아|김형진|terea|텔)', r'\👻', result)
    result = re.sub(r'(해나|혜나|헤나|다혜신|곤뇽\.|곤뇽)', r'\1🦖', result)
    result = re.sub(r'(제시|제씨|졔씨|졔시)', r'\1🥝', result)
    result = re.sub(r'(티미|티미우|timmy)', r'\1™️', result)
    result = re.sub(r'(스티브|steve)', r'\1🍺', result)

    # 3. "하이" → "햄하" 변환 (새로 추가)
    result = re.sub(r'하이', '햄하', result)

    result = re.sub(r'사람들', '햄찌들', result)

    result = re.sub(r'사람이', '햄스터가', result)  # "사람이" → "햄스터가"
    result = re.sub(r'사람을', '햄스터를', result)  # "사람을" → "햄스터는"
    result = re.sub(r'사람이야', '햄스터얌', result)  # "사람이야" → "햄스터얌"
    result = re.sub(r'나는\s*(\S+?)야', '나는 햄스터얌', result)  # "나는 [한글]야" → "나는 햄스터얌"
    result = re.sub(r'햄스터', r'햄스터🐹', result)  # "햄스터" → "햄스터🐹"
    result = re.sub(r'([가-힣])야(?=\s|$|[!?.,])', r'\1얌', result) # "야" → "얌"

    result = re.sub(r'졸리다\b', '졸리다쮸우우..', result)
    result = re.sub(r'잠온다\b', '잠와쮸우우..', result)
    result = re.sub(r'졸려\b', '졸려쮸우우..', result)
    result = re.sub(r'더럽다\b', '더럽다쮸우우..', result)
    result = re.sub(r'더러워', '더러워쮸우우...', result)
  
    result = re.sub(r'(\s|^)해(\s|$)', r'\1해쮸\2', result)
    # 배고픔 표현
 
    result = re.sub(r'배고파요?', '배고파쮸우우...', result)
    result = re.sub(r'배고프다\b', '배고파쮸우우...', result)
    

    # 슬픔 표현  
    result = re.sub(r'슬퍼', '슬퍼쮸우우...', result)
    result = re.sub(r'슬프다\b', '슬퍼쮸우우...', result)

    # 심심함 표현
    result = re.sub(r'심심해', '심심해쮸우우...', result)
    result = re.sub(r'심심하다\b', '심심하다쮸우우...', result)

    result = re.sub(r'냐(멍|개|옹|왈)', '냐쮸', result)
    result = re.sub(r'다(옹|멍|개|왈)', '다쮸', result)
    result = re.sub(r'다\b', '다쮸', result)
    result = re.sub(r'요\b', '요쮸', result)

    # 5. 대답 변환: "응" → "웅", "네" → "넹", "예" → "녱" (제한적)
    result = re.sub(r'^응(?=[!?\s.,]|$)', '웅', result)
    result = re.sub(r'(\s)응(?=[!?\s.,]|$)', r'\1웅', result)
    # "네"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^네([!?.,])', r'넹\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^네(?=\s*$)', '넹', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)네([!?.,])', r'\1넹\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)네(?=\s*$)', r'\1넹', result)  # 중간에 단독으로 끝나는 경우
    # "예"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^예([!?.,])', r'녱\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^예(?=\s*$)', '녱', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)예([!?.,])', r'\1녱\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)예(?=\s*$)', r'\1녱', result)  # 중간에 단독으로 끝나는 경우
    
    # 6. 감탄사 변환: 문장 맨 앞의 감탄사 변환 (문장 끝 처리 전에 실행)
    result = re.sub(r'^와(?=\s|$|[.!?,:;~])', '꾸앙', result)
    result = re.sub(r'^오(?=\s|$|[.!?,:;~])', '끄오', result)
    result = re.sub(r'^아(?=\s|$|[.!?,:;~])', '뀨아', result)
    
    # 7. 감탄사 변환 (앗, 앙, 으악, 아악) - 위치에 관계없이 모두 변환
    result = re.sub(r'(?<![가-힣])앙(?![가-힣])', '찍', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])앗(?![가-힣])', '찍', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])으악(?![가-힣])', '뀨앙', result)  # 앞뒤에 한글이 없는 경우
    result = re.sub(r'(?<![가-힣])아악(?![가-힣])', '끄앙', result)  # 앞뒤에 한글이 없는 경우
    
    # 8. 자음 조합 변환 (긴 패턴부터 먼저 처리)
    result = re.sub(r'(ㅎㅇㅌ|화이팅|파이팅)', '햄이팅', result)  # ㅎㅇ보다 먼저 처리
    result = re.sub(r'(ㅎㅇ|하이)', '햄하', result)
    result = re.sub(r'ㅇㅁ', '어머찍', result)
    result = re.sub(r'ㅁㅇ', '모야찍', result)
    result = re.sub(r'ㄱㅊ', '괜찮찍', result)
    result = re.sub(r'ㄱㄱ', '고고레쮸고!', result)
    result = re.sub(r'ㅅㄱ','수고해라츄우~', result)
    result = re.sub(r'ㅋㅋ+', r'\g<0>햄하하', result)
    result = re.sub(r'ㅎㅎ+', r'\g<0>헤헤헷~', result)
    result = re.sub(r'[ㅜㅠ]+', '\g<0>츄우우..', result)

    # ㄱㅇㅇ를 임시로 보호
    result = re.sub(r'ㄱㅇㅇ', 'TEMP_GYY', result)
    # ㅇㅇ 변환
    result = re.sub(r'ㅇㅇ', '웅찍', result)
    # ㄱㅇㅇ 복원
    result = re.sub(r'TEMP_GYY', 'ㄱㅇㅇ', result)
    
    result = re.sub(r'ㅇㄸ', '어떠햄', result)
    result = re.sub(r'(?<![가-힣])아하(?![가-힣])', '아하쮸', result)  # 앞뒤에 한글이 없는 독립된 "아하"만
    
    # 9. 특별 단어/어절 처리
    # "개웃" → "햄웃" (햄웃겨, 햄웃기다, 햄웃김 등)
    result = re.sub(r'개웃', '햄웃', result)
    
    # 공백 뒤 강조 표현: "햄이쁘", "햄귀엽", "햄귀여" → "햄이쁘", "햄귀엽", "햄귀여" (강조 용법만)
    result = re.sub(r'(\s)개(이쁘|귀엽|귀여)', r'\1햄\2', result)  # 공백 뒤에만
    
    # "존" 강조 표현 변환 (공백 뒤에만)
    result = re.sub(r'(\s)존(잼|맛|맛탱|예|귀|좋)', r'\1햄\2', result)  # 공백 뒤에만
    
    # 뒤에 한글이 오지 않는 경우에만 멍 붙이기
    result = re.sub(r'(맞아|마자|마좌|마쟈)(?![가-힣])', r'\1쮸', result)
   


    #  "했지", "었지", "았지" 변환
    result = re.sub(r'했지', '해찌', result)
    result = re.sub(r'었지', '어찌', result) 
    result = re.sub(r'았지', '아찌', result)
    result = re.sub(r'있지', '이찌', result)
    result = re.sub(r'없지', '업찌', result)

    # 10. 문장 끝에 "찍" 추가 (한국어가 포함된 경우만)
    if re.search(r'[가-힣]', result):
        # 문장 끝 처리 - 문장부호로 끝나는 경우
        result = re.sub(r'([가-힣])(?<![찍쮸찌])(\s*[.!?~\\;]+)(?![가-힣])', r'\1찍\2', result)

        
        # 이모티콘으로 끝나는 경우
        result = re.sub(r'([가-힣])(?<![찍쮸찌])(\s*\^\^\s*$)', r'\1찍\2', result)  # ^^ 이모티콘
        result = re.sub(r'([가-힣])(?<![찍쮸찌])(\s*:\)\s*$)', r'\1찍\2', result)  # :) 이모티콘
        result = re.sub(r'([가-힣])(?<![찍쮸찌])(\s*[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+\s*$)', r'\1찍\2', result)  # 유니코드 이모티콘
        
        # 자음/모음(ㅋㅋ, ㅎㅎ, ㅇㅋ 등) 앞의 한글에 "찍" 추가
        result = re.sub(r'([가-힣])(?<!찍)(\s*[ㄱ-ㅎㅏ-ㅣㅋㅎㅇㅋ]+)', r'\1찍\2', result)  # 자음/모음 앞
        
        # 줄바꿈 처리
        result = re.sub(r'([가-힣])(?<!찍)(\s*\r?\n)', r'\1찍\2', result)  # 줄바꿈으로 끝
        
        # 문장부호 없이 끝나는 경우
        result = re.sub(r'([가-힣])(?<!찍)(\s*$)', r'\1찍\2', result)  # 그냥 끝나는 경우
    
    
    # 12. 불필요한 "찍" 제거 (특별 변환 후 붙은 찍 정리)
    # 단일 패턴 뒤의 찍 제거
    result = re.sub(r'(아하|우하하|하하|해보(바보)|소피|곤뇽|햄하|미스코리냥|#미스코리냥|찍찍|아하쮸|왕왕|어찌|이찌|화이찡|꾸앙|끄오|뀨악|햄잼|햄맛|햄맛탱|햄예|햄귀|햄좋|쮸|어떠햄|뀨앙|쮸우우|햄바|햄빠|햄찌들|햄이팅|미야옹즈|미야옹|햄하하)찍', r'\1', result)
    

    # 13. "아아" 복원
    result = re.sub(r'TEMP_AA', '아아', result)
    
    # 14. 작은따옴표 내용 복원 (맨 마지막에)
    for placeholder, original in quoted_parts.items():
        result = result.replace(placeholder, original)

    return result