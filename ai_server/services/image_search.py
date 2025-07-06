import requests
import torch
import numpy as np
from PIL import Image
from io import BytesIO
from transformers import CLIPProcessor, CLIPModel
import chromadb
from typing import List, Optional, Dict
import logging
import os
import threading
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)

class ImageSearchService:
    """이미지 유사도 검색 서비스"""
    
    def __init__(self, db_base_path: str = "./image_embeddings_db"):
        """
        초기화
        
        Args:
            db_base_path: ChromaDB 기본 저장 경로
        """
        self.db_base_path = db_base_path
        self.model: Optional[CLIPModel] = None
        self.processor: Optional[CLIPProcessor] = None
        self.collections: Dict[str, any] = {}
        self.clients: Dict[str, any] = {}  # ChromaDB 클라이언트 캐시
        self._initialization_lock = threading.Lock()
        self._initialized_animals = set()
        
        # HTTP 세션 설정 (연결 풀링)
        self.session = requests.Session()
        retry_strategy = Retry(
            total=2,  # 최대 2회 재시도
            backoff_factor=0.5,  # 재시도 간격
            status_forcelist=[429, 500, 502, 503, 504],  # 재시도할 HTTP 상태 코드
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # CLIP 모델 초기화 (즉시 로드)
        self._initialize_clip_model()
    
    def _initialize_clip_model(self):
        """CLIP 모델 초기화 (한 번만 실행)"""
        try:
            if self.model is None:
                logger.info("🚀 Loading CLIP model...")
                self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
                self.model.eval()
                logger.info("✅ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"❌ CLIP model initialization failed: {e}")
            raise
    
    def _ensure_chromadb_initialized(self, animal_type: str):
        """특정 동물 타입의 ChromaDB 초기화 (지연 초기화 + 중복 방지)"""
        if animal_type in self._initialized_animals:
            return  # 이미 초기화됨
        
        with self._initialization_lock:
            # 더블 체크 락킹 패턴
            if animal_type in self._initialized_animals:
                return
            
            try:
                logger.info(f"🔄 Initializing ChromaDB for {animal_type}...")
                db_path = f"{self.db_base_path}/{animal_type}_db"
                
                # 디렉토리가 없으면 생성
                os.makedirs(db_path, exist_ok=True)
                
                # 클라이언트 생성 및 캐시
                client = chromadb.PersistentClient(path=db_path)
                self.clients[animal_type] = client
                
                # 컬렉션 가져오기 (없으면 생성)
                collection_name = f"{animal_type}_images"
                try:
                    collection = client.get_collection(collection_name)
                    logger.info(f"📁 Found existing {animal_type} collection")
                except ValueError:
                    # 컬렉션이 없으면 생성
                    collection = client.create_collection(
                        name=collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    logger.info(f"🆕 Created new {animal_type} collection")
                
                self.collections[animal_type] = collection
                self._initialized_animals.add(animal_type)
                logger.info(f"✅ {animal_type.capitalize()} ChromaDB initialized")
                
            except Exception as e:
                logger.error(f"❌ ChromaDB initialization failed for {animal_type}: {e}")
                raise

    def download_image_from_url(self, image_url: str) -> Image.Image:
        """웹 URL에서 이미지 다운로드 (최적화된 타임아웃 및 재시도)"""
        try:
            # 세션을 사용하여 연결 풀링 활용, 타임아웃 5초로 단축
            response = self.session.get(
                image_url, 
                timeout=5,  # 10초 → 5초로 단축
                verify=True,  # SSL 검증 활성화
                headers={
                    'User-Agent': 'Mozilla/5.0 (compatible; ImageSearchBot/1.0)'
                }
            )
            response.raise_for_status()
            
            # 이미지 크기 체크 (10MB 제한)
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 10 * 1024 * 1024:
                raise ValueError("이미지 파일이 너무 큽니다 (10MB 초과)")
            
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # 이미지 크기 검증
            if image.size[0] < 32 or image.size[1] < 32:
                raise ValueError("이미지가 너무 작습니다 (최소 32x32)")
            
            logger.debug(f"✅ Image downloaded: {image.size}")
            return image
            
        except requests.exceptions.Timeout:
            logger.warning(f"⏰ Image download timeout: {image_url}")
            raise ValueError("이미지 다운로드 시간 초과 (5초)")
        except requests.exceptions.SSLError:
            logger.warning(f"🔒 SSL verification failed: {image_url}")
            raise ValueError("SSL 인증서 검증 실패")
        except requests.exceptions.RequestException as e:
            logger.error(f"🌐 Network error downloading image: {e}")
            raise ValueError(f"이미지 다운로드 네트워크 오류: {e}")
        except Exception as e:
            logger.error(f"❌ Image download failed: {e}")
            raise ValueError(f"이미지 다운로드 실패: {e}")

    def extract_query_embedding(self, image: Image.Image) -> np.ndarray:
        """쿼리 이미지에서 CLIP 임베딩 추출"""
        try:
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                outputs = self.model.get_image_features(**inputs)
                embedding = outputs.cpu().numpy().squeeze()
            
            # 임베딩 정규화
            embedding = embedding / np.linalg.norm(embedding)
            
            return embedding
        except Exception as e:
            logger.error(f"❌ Embedding extraction failed: {e}")
            raise

    def search_chromadb(self, query_embedding: np.ndarray, animal_type: str, n_results: int = 3) -> List[str]:
        """ChromaDB에서 유사도 검색"""
        try:
            # 동물 타입 검증
            if animal_type not in ["cat", "dog"]:
                raise ValueError(f"지원하지 않는 동물 타입: {animal_type}")
            
            # ChromaDB 지연 초기화
            self._ensure_chromadb_initialized(animal_type)
            
            collection = self.collections[animal_type]
            results = collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results
            )
            
            image_urls = results['ids'][0] if results['ids'] else []
            return image_urls
            
        except Exception as e:
            logger.error(f"❌ ChromaDB search failed: {e}")
            raise

    def search_similar_images(self, image_url: str, animal_type: str, n_results: int = 3) -> List[str]:
        """
        메인 함수: 이미지 URL을 받아서 유사한 이미지 URL 반환
        
        Args:
            image_url (str): 검색할 이미지의 웹 URL
            animal_type (str): 동물 종류 ("cat" 또는 "dog")
            n_results (int): 반환할 결과 개수 (기본값: 3)
            
        Returns:
            List[str]: 유사한 이미지 URL 리스트
        """
        start_time = time.time()
        
        try:
            # 동물 타입 검증
            if animal_type not in ["cat", "dog"]:
                raise ValueError("동물 타입은 'cat' 또는 'dog'여야 합니다")
            
            # 이미지 다운로드 및 임베딩 추출
            image = self.download_image_from_url(image_url)
            query_embedding = self.extract_query_embedding(image)
            
            # 유사도 검색
            similar_urls = self.search_chromadb(query_embedding, animal_type, n_results)
            
            elapsed_time = time.time() - start_time
            logger.info(f"✅ Found {len(similar_urls)} similar images for {animal_type} in {elapsed_time:.2f}s")
            return similar_urls
            
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(f"❌ Image search failed in {elapsed_time:.2f}s: {e}")
            raise
    
    def __del__(self):
        """소멸자: HTTP 세션 정리"""
        if hasattr(self, 'session'):
            self.session.close()


# 전역 인스턴스 (스레드 안전 싱글톤 패턴)
_image_search_service = None
_service_lock = threading.Lock()

def get_image_search_service() -> ImageSearchService:
    """이미지 검색 서비스 인스턴스 반환 (스레드 안전 싱글톤)"""
    global _image_search_service
    
    if _image_search_service is None:
        with _service_lock:
            # 더블 체크 락킹 패턴
            if _image_search_service is None:
                _image_search_service = ImageSearchService()
    
    return _image_search_service 