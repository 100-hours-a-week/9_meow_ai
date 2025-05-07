from typing import List, Optional
import asyncio
import ast
import os
from dotenv import load_dotenv

class APIKeyPool:
    def __init__(self, api_keys: List[str]):
        self.api_keys = api_keys
        self.current_index = 0
        self.lock = asyncio.Lock()

    async def get_available_key(self) -> Optional[str]:
        async with self.lock:
            if not self.api_keys:
                return None
            key = self.api_keys[self.current_index]
            self.current_index = (self.current_index + 1) % len(self.api_keys)
            print(f"선택된 키: key_{self.current_index + 1}")
            return key

    def get_available_key_count(self) -> int:
        return len(self.api_keys)

def initialize_key_pool() -> APIKeyPool:
    load_dotenv()
    api_keys = []

    key_env = os.getenv("GOOGLE_API_KEYS_list") or os.getenv("GOOGLE_API_KEYS")

    if key_env:
        try:
            parsed_keys = ast.literal_eval(key_env)
            if isinstance(parsed_keys, list):
                api_keys = parsed_keys
                print(f"Loaded {len(api_keys)} API key(s).")
            else:
                api_keys = [parsed_keys]
                print("Loaded 1 API key.")
        except (ValueError, SyntaxError) as e:
            print(f"Warning: Failed to parse API keys: {e}")
    else:
        print("Warning: No API keys found. Please set GOOGLE_API_KEYS or GOOGLE_API_KEYS_list")

    return APIKeyPool(api_keys=api_keys)
