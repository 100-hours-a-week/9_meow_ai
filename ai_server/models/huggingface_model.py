"""
허깅페이스 Meow-HyperCLOVAX 풀파인튜닝 모델 구현 (T4 GPU 최적화 버전)
"""

import logging
import os
import torch
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
from ai_server.config import get_settings
# import wandb

# 로깅 설정
logger = logging.getLogger(__name__)

class PostModel:
    """
    풀파인튜닝된 Meow-HyperCLOVAX 모델을 로드하고 관리하는 클래스 (T4 GPU 전용)
    """
    # 풀파인튜닝 모델 경로 및 파라미터 정의
    FINE_TUNED_MODEL_PATH = "haebo/Meow-HyperCLOVAX-1.5B_FullFT_fp32_0527f"
    MODEL_LOAD_DTYPE = torch.float32
    MODEL_MAX_LENGTH = 1024 # HyperCLOVAX-SEED-1.5B(1/2 값)
    MODEL_MAX_NEW_TOKENS = 200 
    MODEL_TEMPERATURE = 0.2
    MODEL_TOP_P = 0.3 # 좀 더 낮출 필요 있을 듯 (사용할 확률 분포값)
    MODEL_REPETITION_PENALTY = 1.6 # 반복 억제 
    
    def __init__(self):
        """
        풀파인튜닝 모델 초기화
        """
        # accelerate 관련 환경 변수 설정 (None 오류 방지)
        os.environ.setdefault("ACCELERATE_USE_CPU", "false")
        os.environ.setdefault("ACCELERATE_USE_DEEPSPEED", "false")
        os.environ.setdefault("ACCELERATE_USE_FSDP", "false")
        
        # 모델 경로 설정
        self.fine_tuned_model_path = self.FINE_TUNED_MODEL_PATH
        self.model_load_dtype = self.MODEL_LOAD_DTYPE
        
        # 인증 토큰은 환경 변수에서 가져옴
        settings = get_settings()
        self.auth_token = settings.HUGGINGFACE_TOKEN
        
        # GPU 환경 최적화 설정
        self._setup_gpu()
        
        logger.info(f"파인튜닝 모델 로드 시작: {self.fine_tuned_model_path}")
        
        # 풀파인튜닝 모델 로드
        self._load_fine_tuned_model()
        
        # 생성 설정 저장
        self.max_new_tokens = self.MODEL_MAX_NEW_TOKENS
        self.temperature = self.MODEL_TEMPERATURE
        self.top_p = self.MODEL_TOP_P
        self.repetition_penalty = self.MODEL_REPETITION_PENALTY
        
        logger.info(f"파인튜닝 모델 로드 완료: {self.fine_tuned_model_path}")
        self._initialized = True

        # # W&B 실험 추적 초기화
        # wandb.init(
        #     project="meow-hyperclovax",
        #     name="FullFT-v0527-0611",  # 실험 이름
        #     config={
        #         "model_path": self.FINE_TUNED_MODEL_PATH,
        #         "max_new_tokens": self.MODEL_MAX_NEW_TOKENS,
        #         "temperature": self.MODEL_TEMPERATURE,
        #         "top_p": self.MODEL_TOP_P,
        #         "repetition_penalty": self.MODEL_REPETITION_PENALTY,
        #         "dtype": str(self.MODEL_LOAD_DTYPE),
        #     }
        # )
    
    def _setup_gpu(self):
        """
        GPU 설정 초기화 및 최적화
        """
        # GPU 메모리 분할 크기 최적화
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
        
        # CUDA 메모리 캐시 설정
        torch.cuda.empty_cache()
        
        # GPU 메모리 정보 로깅
        mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        mem_name = torch.cuda.get_device_name(0)
        logger.info(f"GPU: {mem_name}, 메모리: 총 {mem_total:.2f}GB")
        logger.info("GPU 초기화 완료")
    
    def _load_fine_tuned_model(self):
        """
        파인튜닝된 모델을 직접 로드
        """
        # 풀파인튜닝된 모델 직접 로드 (32비트)
        logger.info(f"파인튜닝 모델 로드 중: {self.fine_tuned_model_path}")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.fine_tuned_model_path,
            device_map="auto",  # 자동 디바이스 매핑
            torch_dtype=self.model_load_dtype,  # 32비트 정밀도
            token=self.auth_token,
            low_cpu_mem_usage=True,
            trust_remote_code=True
        )
        
        # 토크나이저 로드
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.fine_tuned_model_path,
            token=self.auth_token,
            model_max_length=self.MODEL_MAX_LENGTH,
            trust_remote_code=True
        )
        
        # 패딩 토큰 설정 (필요한 경우)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 모델을 평가 모드로 설정
        self.model.eval()
        
        logger.info("풀파인튜닝 모델과 토크나이저 로드 완료")
    
    def _postprocess(self, text: str, original_content: str = "") -> str:
        import re
        from collections import Counter

        EMOJI_PATTERN = (
            "[" +
            "\U0001F600-\U0001F64F"
            "\U0001F300-\U0001F5FF"
            "\U0001F680-\U0001F6FF"
            "\U0001F1E0-\U0001F1FF"
            "\U00002700-\U000027BF"
            "\U0001F900-\U0001F9FF"
            "\U00002600-\U000026FF"
            "]"
        )

        # transformed_content 포맷 제거
        text = re.sub(r"### transformed_content:\s*", "", text).strip()

        # 줄바꿈 및 백슬래시 제거
        text = re.sub(r'(\\r\\n|\\r|\\n|\r|\n)', '', text)

        # 해시태그 제거
        text = re.sub(r'#\S+', '', text)

        # 이모지 연속 제거
        text = re.sub(f"({EMOJI_PATTERN}){EMOJI_PATTERN}+", r"\1", text)

        # 이모지 2개 초과 시 앞의 2개만 남기고 제거
        emojis = re.findall(EMOJI_PATTERN, text)
        if len(emojis) > 2:
            keep = emojis[:2]
            text = re.sub(EMOJI_PATTERN, '', text) + ''.join(keep)

        # 기호나 감탄사 반복 (ex. ! ! ! ...) 압축
        text = re.sub(r'([!?\.💢❤⭐✨🐾…]{1})( \1|\1){2,}', r'\1\1', text)

        # 특수문자/비정상 문자 제거
        text = re.sub(r"[️‹›／]", '', text)

        # 동일 단어 반복 축소
        words = re.findall(r'\b\w+\b', text)
        counts = Counter(words)
        for word, count in counts.items():
            if count >= 4:
                text = re.sub(rf'\b({re.escape(word)})\b', '', text, count - 2)

        # 금지어 제거
        for word in ['system', '안올라간다']:
            text = text.replace(word, '')

        # 비정상적으로 끊긴 문장 정리
        text = re.sub(r'([가-힣a-zA-Z])\s*\.+\s*$', r'\1.', text)
        text = re.sub(r'\.\.+', '.', text)

        # 너무 짧거나 의미 없는 문장 제거
        if len(text) < 5 or re.fullmatch(r'[\W\d\s]+', text):
            return "[출력 오류] 결과 생성이 실패했어요."

        # 길이 제한
        if original_content:
            # 길이 조건 별 최대 길이 설정
            original_len = len(original_content)
            if original_len <= 30:
                max_len = int(3.0 * original_len)
            else: 
                max_len = int(2.0 * original_len)

            if len(text) > max_len:
                # 공백 기준으로 자르고, 초과 부분을 버림
                words = text.split()
                trimmed_text = ""
                for word in words:
                    if len(trimmed_text) + len(word) + 1 > max_len:
                        break
                    trimmed_text += word + " "
                text = trimmed_text.strip()

        return re.sub(r'\s+', ' ', text).strip()


    async def generate(
        self, 
        prompt: str, 
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        original_content: Optional[str] = str,
        **kwargs
    ) -> str:
        """
        프롬프트를 기반으로 텍스트 생성
        
        Args:
            prompt: 입력 프롬프트
            max_new_tokens: 최대 생성 토큰 수
            temperature: 생성 온도 (창의성 조절)
            top_p: nucleus sampling 파라미터
            **kwargs: 추가 생성 파라미터
            
        Returns:
            생성된 텍스트
        """
        # 생성 파라미터 설정
        # max_new_tokens = max_new_tokens or self.max_new_tokens
        temperature = temperature or self.temperature
        top_p = top_p or self.top_p
        repetition_penalty = repetition_penalty or self.repetition_penalty

        # 입력 토큰화
        inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        max_new_tokens = max_new_tokens or self.max_new_tokens
        
        # 어텐션 마스크 생성
        attention_mask = torch.ones_like(inputs)
        # 텍스트 생성
        with torch.no_grad():
            outputs = self.model.generate(
                inputs,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                temperature=temperature, # 낮게 설정 : 높은 확률의 토큰 위주 선택
                do_sample=True, # True + 낮은 Temp : 약간의 다양성 부여 (False 하면 가장 높은 확률의 결과만 도출)
                top_p=top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,  
                **kwargs
            )

        # 새로 생성된 토큰만 디코딩 (입력 제외)
        new_tokens = outputs[0][inputs.shape[1]:]
        decoded = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        processed_text = self._postprocess(decoded, original_content)

        # # W&B 로그 기록
        # wandb.log({
        #     "prompt": prompt,
        #     "generated_text": processed_text,
        #     "output_length": len(self.tokenizer.encode(processed_text)),
        #     "temperature": temperature,
        #     "top_p": top_p,
        #     "repetition_penalty": repetition_penalty
        # })
        # print("processed_text:", processed_text)
        return processed_text
    
    def get_model_info(self) -> dict:
        """
        모델 정보 반환
        
        Returns:
            모델 정보 딕셔너리
        """
        return {
            "model_path": self.fine_tuned_model_path,
            "max_length": self.model_max_length,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "device": str(self.model.device) if hasattr(self, 'model') else "not_loaded",
            "initialized": getattr(self, '_initialized', False),
            "repetition_penalty" : self.repetition_penalty,
        }
    
    def unload(self):
        """모델 메모리에서 언로드"""
        if hasattr(self, "model"):
            del self.model
            del self.tokenizer
            self._initialized = False
            torch.cuda.empty_cache()
            logger.info("풀파인튜닝 모델 언로드 완료") 