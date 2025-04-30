import asyncio
import os
import ast
import time
from dotenv import load_dotenv
from ai_server.key_pool import APIKeyPool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from ai_server.schemas import Emotion, PostType
from ai_server.templates import DYNAMIC_PROMPT, ANIMAL_TRAITS

class NoAvailableKeyError(Exception):
    """사용 가능한 API 키가 없을 때 발생하는 예외"""
    def __init__(self, message: str, available_keys: int = 0):
        self.message = message
        self.available_keys = available_keys
        super().__init__(f"{message} (사용 가능한 키: {available_keys}개)")

async def example_usage():
    load_dotenv("/Users/jaeseoksee/Documents/code/meong/haebo/.env")
    raw_keys = os.getenv("GOOGLE_API_KEYS")
    api_keys = ast.literal_eval(raw_keys)

    key_pool = APIKeyPool(api_keys=api_keys, max_requests_per_min=5)

    # 요청 카운터 리셋 태스크 실행
    async def reset_counters_periodically():
        while True:
            await asyncio.sleep(60)
            print("🔄 요청 카운터 리셋됨")
            await key_pool.reset_counters()

    asyncio.create_task(reset_counters_periodically())

    # 프롬프트 템플릿 설정
    prompt = PromptTemplate(
        template=DYNAMIC_PROMPT,
        input_variables=["content", "emotion", "animal_type", "suffix", "characteristics"]
    )

    # 모의 API 요청 함수
    async def make_api_request(i):
        while True:  # 사용 가능한 키가 생길 때까지 대기
            api_key = await key_pool.get_available_key()
            if not api_key:
                available_keys = key_pool.get_available_key_count()
                if available_keys == 0:
                    print(f"⚠️ 경고: 사용 가능한 API 키가 없습니다. (요청 {i} 대기)")
                    await asyncio.sleep(1)  # 1초 대기 후 재시도
                    continue
                else:
                    print(f"⚠️ 경고: 현재 키 사용량 초과. {available_keys}개의 키가 사용 가능합니다. (요청 {i} 대기)")
                    await asyncio.sleep(1)  # 1초 대기 후 재시도
                    continue
            
            try:
                start = time.time()
                print(f"[요청 {i}] ✅ 요청 시작 - 시간: {start:.2f}")
                
                # LLM 인스턴스 생성
                llm = ChatGoogleGenerativeAI(
                    model="gemini-2.0-flash",
                    temperature=0.7,
                    convert_system_message_to_human=True,
                    google_api_key=api_key
                )
                
                # 체인 생성
                chain = LLMChain(llm=llm, prompt=prompt)
                
                # 테스트용 입력 데이터
                test_content = "안녕하세요, 반갑습니다!"
                test_emotion = Emotion.HAPPY
                test_post_type = PostType.CAT
                
                # 동물 특성 가져오기
                animal_traits = ANIMAL_TRAITS[test_post_type.value][test_emotion.value]
                
                # 변환 실행
                result = await chain.ainvoke({
                    "content": test_content,
                    "emotion": test_emotion.value,
                    "animal_type": test_post_type.value,
                    "suffix": animal_traits["suffix"],
                    "characteristics": ", ".join(animal_traits["characteristics"])
                })
                
                end = time.time()
                print(f"[결과 {i}] ✅ {end - start:.2f}초 / 결과: {result['text']}")
                break  # 요청 성공 시 루프 종료
                
            except Exception as e:
                print(f"[{i}] ❗ 요청 오류: {e}")
                await asyncio.sleep(1)  # 오류 발생 시 1초 대기 후 재시도
            finally:
                if api_key:
                    await key_pool.release_key(api_key)

    async def run_requests_with_retry(request_count: int, start_index: int = 0):
        tasks = []
        for i in range(request_count):
            task = asyncio.create_task(make_api_request(i + start_index))
            tasks.append(task)
            
        try:
            # 모든 태스크 완료 대기
            await asyncio.gather(*tasks)
            return True
            
        except Exception as e:
            print(f"실행 중 오류 발생: {e}")
            return False

    # 1차 요청 실행
    print("\n📝 1차 요청 시작")
    i = 0
    k = 100
    while True:
        result = await run_requests_with_retry(4)
        i += 1
        if not result:
            print("\n⏳ 요청 실패. 잠시 후 재시도합니다...")
            await asyncio.sleep(1)
            continue
            
        # 총 요청이 k개에 도달하면 종료 
        if i >= k:
            print(f"\n✅ 총 {k}개의 요청이 완료되어 종료합니다.")
            break
            
        # 다음 요청 전 1초 대기
        await asyncio.sleep(1)
    return

if __name__ == "__main__":
    asyncio.run(example_usage())
