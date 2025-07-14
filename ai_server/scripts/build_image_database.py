"""
이미지 데이터베이스 구축 스크립트
data 디렉토리의 URL 파일들을 읽어서 ChromaDB에 저장
"""
import os
import sys
import logging
from pathlib import Path

# ChromaDB telemetry 비활성화 (hang 방지)
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# 프로젝트 루트 경로를 Python path에 추가
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ai_server.model.image_search import ImageSearchService

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def build_database():
    """이미지 데이터베이스 구축"""
    service = None
    try:
        # 이미지 검색 서비스 초기화
        logger.info("이미지 검색 서비스 초기화 중...")
        service = ImageSearchService()
        
        # data 디렉토리 경로
        data_dir = project_root / "data"
        
        # ChromaDB 컬렉션 먼저 초기화
        logger.info("ChromaDB 컬렉션 초기화 중...")
        service._ensure_chromadb_initialized("cat")
        service._ensure_chromadb_initialized("dog")
        
        # 고양이 이미지 URL 파일 처리
        cat_url_file = data_dir / "cat_image_url.txt"
        if cat_url_file.exists():
            logger.info("고양이 이미지 DB 구축 시작...")
            with open(cat_url_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    url = line.strip()
                    if url:  # 빈 줄 건너뛰기
                        try:
                            # 이미지 다운로드 및 임베딩 추출
                            image = service.download_image_from_url(url)
                            embedding = service.extract_query_embedding(image)
                            
                            # ChromaDB에 저장
                            service.collections["cat"].add(
                                ids=[url],
                                embeddings=[embedding.tolist()]
                            )
                            logger.info(f"[{i}] 고양이 이미지 추가 완료: {url[:50]}...")
                            
                        except Exception as e:
                            logger.error(f"[{i}] 고양이 이미지 추가 실패: {url[:50]}... - {e}")
        
        # 강아지 이미지 URL 파일 처리
        dog_url_file = data_dir / "dog_image_url.txt"
        if dog_url_file.exists():
            logger.info("강아지 이미지 DB 구축 시작...")
            with open(dog_url_file, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f, 1):
                    url = line.strip()
                    if url:  # 빈 줄 건너뛰기
                        try:
                            # 이미지 다운로드 및 임베딩 추출
                            image = service.download_image_from_url(url)
                            embedding = service.extract_query_embedding(image)
                            
                            # ChromaDB에 저장
                            service.collections["dog"].add(
                                ids=[url],
                                embeddings=[embedding.tolist()]
                            )
                            logger.info(f"[{i}] 강아지 이미지 추가 완료: {url[:50]}...")
                            
                        except Exception as e:
                            logger.error(f"[{i}] 강아지 이미지 추가 실패: {url[:50]}... - {e}")
        
        # 결과 출력
        cat_count = service.collections["cat"].count()
        dog_count = service.collections["dog"].count()
        
        logger.info("이미지 데이터베이스 구축 완료!")
        logger.info(f"통계:")
        logger.info(f"   - 고양이 이미지: {cat_count}개")
        logger.info(f"   - 강아지 이미지: {dog_count}개")
        logger.info(f"   - 총 이미지: {cat_count + dog_count}개")
        
    except Exception as e:
        logger.error(f"데이터베이스 구축 실패: {e}")
        raise
    finally:
        # 명시적 리소스 정리
        if service:
            logger.info("리소스 정리 시작...")
            service.cleanup()
            logger.info("리소스 정리 완료")

if __name__ == "__main__":
    build_database()
    # 강제 종료를 위한 추가 정리
    import gc
    gc.collect()
    logger.info("스크립트 종료") 