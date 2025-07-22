import re

def monkey_converter(text):
    """텍스트를 몽키체로 변환하는 함수"""

    if not text or not isinstance(text, str):
        return text
    
    result = text
   
    # 영어 문장 체크 (알파벳이 최소 1개 이상 포함되고, 알파벳, 공백, 숫자, 기본 문장부호만 포함)
    if re.match(r'^[a-zA-Z\s\d.,!?;:\'"-]+$', result.strip()) and re.search(r'[a-zA-Z]', result.strip()):

        # 문장 끝에 문장부호가 있는 경우 앞에 meow 추가
        if re.search(r'[.!?]$', result.strip()):
            result = re.sub(r'([.!?])$', r' woo kki kki🐒🍌\1', result.strip())
        else:
            # 문장부호가 없는 경우 그냥 meow 추가
            result = result.strip() + ' woo kki kki🐒🍌'
        return result

    # 0. 작은따옴표 안의 내용 보호
    quoted_parts = {}
    quote_pattern = r"'([^']*?)'"
    
    def replace_quoted(match):
        placeholder = f"TEMP_QUOTE_{len(quoted_parts)}"
        quoted_parts[placeholder] = match.group(0)
        return placeholder
    
    result = re.sub(quote_pattern, replace_quoted, result)

    # 미야옹 이스터에그  
    result = re.sub(r'(미야옹즈|미야옹)', r'✨\1✨', result)
    result = re.sub(r'해보\b', '해보(바보)🐈', result)
    result = re.sub(r'소피 바보\b', '소피는 너무 예쁘다❤️', result)
    result = re.sub('소피', '소피🎀', result)
    result = re.sub(r'(테리아|김형진|텔)', r'\1👻', result)
    result = re.sub(r'(티미|티미우|티바시)', r'\1™️', result)
    result = re.sub(r'(스티브|스팁|슽)', r'\1🍺', result)
    result = re.sub(r'(해나|혜나|헤나|다혜신|곤뇽\.|곤뇽)',r'\1🦖', result)
    result = re.sub(r'(제시|제씨|졔씨|졔시)', r'\1🥝', result)
    result = re.sub(r'샌디', r'샌디🐣', result)
    
    # 1. "아아" 보호 (아이스아메리카노, 의성어)
    result = re.sub(r'아아', 'TEMP_AA', result)
    
    # 2. "안녕" → "몽하" 변환
    result = re.sub(r'안녕\b', '몽하', result)

    # 3. "하이" → "몽하" 변환 (새로 추가)
    result = re.sub(r'하이', '몽하', result)
    result = re.sub(r'바이', '몽바', result)
    result = re.sub(r'빠이', '몽빠', result)

    result = re.sub(r'사람들', '숭이들🐵', result)
    
    result = re.sub(r'사람이', '숭이가', result)  # "사람" → "숭이"
    result = re.sub(r'사람을', '숭이를', result)  # "사람을" → "숭이를"
    result = re.sub(r'사람이야', '숭이얌', result)  # "사람이야" → "숭이야"
    result = re.sub(r'나는\s*(\S+?)야', '나는 원숭이야', result)  # "나는 [한글]야" → "나는 숭이야"
    result = re.sub(r'(원숭이)', r'숭이🐵', result)  # "숭이" → "숭이🐵"

     # 자음 조합 변환 (긴 패턴부터 먼저 처리)
    result = re.sub(r'(ㅎㅇㅌ|화이팅|파이팅|홧팅|화팅|파팅|팟팅)', '몽이팅', result)  # ㅎㅇ보다 먼저 처리
    result = re.sub(r'ㅎㅇ', '몽하', result)
    result = re.sub(r'ㅇㅁ', '어머어몽', result)
    result = re.sub(r'ㅁㅇ', '모야끼끼', result)
    result = re.sub(r'ㄱㅊ', '괜찮끼', result)
    result = re.sub(r'(ㄱㄱ|고고)', '레쓰끼끼', result)
    result = re.sub(r'ㅅㄱ', '수고해라끼', result)
    result = re.sub(r'(ㅇㅋ|ㅇㅋㄷㅋ|오키|오키도키|옼돜|오케이)', r'몽키도키', result)

    # ㄱㅇㅇ 귀엽끼
    result = re.sub(r'ㄱㅇㅇ', '귀엽끼', result)
    # ㅇㅇ 변환
    result = re.sub(r'ㅇㅇ', '웅끼끼', result)

    result = re.sub(r'^(와|오|아|우|우아아)(?=\s|$|[.!?,:;])', '우!아!아!', result)

    # 9. 특별 단어/어절 처리
    # "개웃" → "몽웃" (개웃겨, 개웃기다, 개웃김 등)
    result = re.sub(r'개웃기', '몽우끼', result)
    result = re.sub(r'개웃', '몽웃', result)

    # 공백 뒤 강조 표현: "개이쁘", "개귀엽", "개귀여" → "몽이쁘", "몽귀엽", "몽귀여" (강조 용법만)
    result = re.sub(r'(\s)개(이쁘|귀엽|귀여)', r'\1몽\2', result)  # 공백 뒤에만
    
    # "존" 강조 표현 변환 (공백 뒤에만)
    result = re.sub(r'(\s)존(잼|맛|맛탱|예|귀|좋)', r'\1몽\2', result)  # 공백 뒤에만 
    
    # 뒤에 한글이 오지 않는 경우에만 끼끼 붙이기
    result = re.sub(r'(맞아|마자|마좌|마쟈)(?![가-힣])', r'\1끼끼', result)
    
    result = re.sub(r'([가-힣])다\b', r'\1다끼끼', result)
    result = re.sub(r'([가-힣])냐\b', r'\1냐끼끼', result)


    result = re.sub(r'냐(멍|개|옹|왈)', '냐끼끼', result)
    result = re.sub(r'다(옹|멍|개|왈)', '다끼끼', result)
    result = re.sub(r'다\b', '다끼끼', result)
    result = re.sub(r'요\b', '우끼끼', result)
    result = re.sub(r'멍/b', '끼끼', result)
    result = re.sub(r'군/b', '군끼끼', result)
    


    # 5. 대답 변환: "응" → "뭉", "네" → "뭉", "예" → "몡" (제한적)
    result = re.sub(r'^응(?=[!?\s.,]|$)', '뭉', result)
    result = re.sub(r'(\s)응(?=[!?\s.,]|$)', r'\1뭉', result)
    # "네"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^네([!?.,])', r'뭉\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^네(?=\s*$)', '뭉', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)네([!?.,])', r'\1뭉\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)네(?=\s*$)', r'\1뭉', result)  # 중간에 단독으로 끝나는 경우
    # "예"는 명확한 대답일 때만 변환 (문장부호와 함께)
    result = re.sub(r'^예([!?.,])', r'몡\1', result)  # 원본 문장부호 유지
    result = re.sub(r'^예(?=\s*$)', '몡', result)  # 단독으로 끝나는 경우
    result = re.sub(r'(\s)예([!?.,])', r'\1몡\2', result)  # 원본 문장부호 유지
    result = re.sub(r'(\s)예(?=\s*$)', r'\1몡', result)  # 중간에 단독으로 끝나는 경우
    
    # 10. 문장 끝에 "몽" 추가 (한국어가 포함된 경우만)
    if re.search(r'[가-힣]', result):
        # 문장 끝 처리 - 문장부호로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!몽)(\s*[.!?~\\;]+)(?![가-힣])', r'\1끼끼\2', result)

        
        # 이모티콘으로 끝나는 경우
        result = re.sub(r'([가-힣])(?<!몽)(\s*\^\^\s*$)', r'\1끼끼\2', result)  # ^^ 이모티콘
        result = re.sub(r'([가-힣])(?<!몽)(\s*:\)\s*$)', r'\1끼끼\2', result)  # :) 이모티콘
        result = re.sub(r'([가-힣])(?<!몽)(\s*[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\U0001F680-\U0001F6FF\U0001F1E0-\U0001F1FF\u2600-\u26FF\u2700-\u27BF]+\s*$)', r'\1몽\2', result)  # 유니코드 이모티콘
        
        # 자음/모음(ㅋㅋ, ㅎㅎ, ㅇㅋ 등) 앞의 한글에 "몽" 추가
        result = re.sub(r'([가-힣])(?<!몽)(\s*[ㄱ-ㅎㅏ-ㅣㅋㅎㅇㅋ]+)', r'\1끼끼\2', result)  # 자음/모음 앞
        
        # 줄바꿈 처리
        result = re.sub(r'([가-힣])(?<!몽)(\s*\r?\n)', r'\1끼끼\2', result)  # 줄바꿈으로 끝
        
        # 문장부호 없이 끝나는 경우
        result = re.sub(r'([가-힣])(?<!몽)(\s*$)', r'\1끼끼\2', result)  # 그냥 끝나는 경우
    

    # 12. 불필요한 "몽" 제거 (특별 변환 후 붙은 몽 정리)
    # 단일 패턴 뒤의 몽 제거
    result = re.sub(r'(우우|아하|우하하|하하|해보(바보)|소피|곤뇽|끼|뭉|몡|몽|몽이팅|몽하|몽바|몽빠|몽잼|몽맛|몽맛탱|몽예|몽귀|몽좋|우!아!아|미야옹즈|미야옹)끼끼', r'\1', result)
    
    # ㅋ 개수별 변환 (긴 패턴부터 먼저 처리)
    result = re.sub(r'ㅋ{26,}', 'ㄲㄲㄲ크킄킄ㄲㄲ키끼끼우끼킼ㄲ캭캭우컄ㄲㄲㄲㄲㄲㄲㄲㄲ캬', result)
    result = re.sub(r'ㅋ{15,25}', '우키킼ㄲㄲㄲㄲㄲㅌ우끼우낔ㄲㄲㄲㄲ', result)
    result = re.sub(r'ㅋ{8,14}', '우끼우낔ㄲㄲㄲㄲㄲ', result)
    result = re.sub(r'ㅋ{4,7}', '우키킼ㄲㄲㄲ', result)
    result = re.sub(r'ㅋ{3}', '우키킼ㄲ', result)
    result = re.sub(r'ㅋ{2}', '우키키', result)
    result = re.sub(r'ㅋ{1}', '우낔', result)
    result = re.sub(r'[ㅜㅠ]+', r'\g<0>몽..', result)

    
    result = re.sub(r'TEMP_AA', '아아', result)
    # 13. 작은따옴표 안의 내용 복원
    for placeholder, original in quoted_parts.items():
        result = result.replace(placeholder, original)


    return result