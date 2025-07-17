#!/bin/bash

# vLLM 서버 직접 시작 스크립트
# 배포 환경에서 안정적인 실행을 위한 백업 방안

# 공통 유틸리티 함수들 로드
source "$(dirname "$0")/gpu_cache_cleanup.sh"

echo "=== vLLM 직접 시작 스크립트 ==="
echo "모델: ${VLLM_MODEL_PATH:-haebo/meow-clovax-v3}"
echo "포트: ${VLLM_PORT:-8001}"

# 메모리 임계값 설정
MEMORY_THRESHOLD_MB=15000  # 15GB
MAX_RETRIES=3
RETRY_COUNT=0

# PID 파일 경로
VLLM_PID_FILE="/tmp/vllm_server.pid"

# 시작 전 정리 및 체크
echo "=== 시작 전 시스템 체크 ==="

# 1. 기존 프로세스 정리
cleanup_vllm_processes

# 2. PyTorch 캐시 정리
cleanup_pytorch_cache

# 3. GPU 메모리 상태 체크
if ! check_gpu_memory $MEMORY_THRESHOLD_MB; then
    echo "GPU 메모리 상태가 좋지 않습니다. 추가 정리 수행..."
    # 추가 정리 로직
    cleanup_pytorch_cache
    sleep 5
    if ! check_gpu_memory $MEMORY_THRESHOLD_MB; then
        echo "GPU 메모리 정리 실패. 종료합니다."
        exit 1
    fi
fi

# 환경변수 출력
echo "=== vLLM 설정 정보 ==="
echo "GPU 메모리 사용률: ${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
echo "최대 모델 길이: ${VLLM_MAX_MODEL_LEN:-512}"
echo "동시 시퀀스 수: ${VLLM_MAX_NUM_SEQS:-4}"
echo "프리픽스 캐싱: ${VLLM_ENABLE_PREFIX_CACHING:-false}"

# vLLM 서버 시작 루프
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "vLLM 서버 시작 시도 ($((RETRY_COUNT + 1))/$MAX_RETRIES)..."
    
    # 시작 전 메모리 체크
    if ! check_gpu_memory $MEMORY_THRESHOLD_MB; then
        echo "시작 전 메모리 체크 실패"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        continue
    fi
    
    # vLLM 서버 시작
    python -m vllm.entrypoints.openai.api_server \
        --model "${VLLM_MODEL_PATH:-haebo/meow-clovax-v3}" \
        --host "${VLLM_HOST:-0.0.0.0}" \
        --port "${VLLM_PORT:-8001}" \
        --served-model-name "${VLLM_SERVED_MODEL_NAME:-meow-clovax-v3}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.4}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-512}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-4}" &
    
    # PID 저장
    VLLM_PID=$!
    echo $VLLM_PID > "$VLLM_PID_FILE"
    
    # 프로세스 시작 확인
    sleep 10
    if ! kill -0 $VLLM_PID 2>/dev/null; then
        echo "vLLM 프로세스가 시작 직후 종료됨"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        rm -f "$VLLM_PID_FILE"
        continue
    fi
    
    # 프로세스가 정상 종료될 때까지 대기
    wait $VLLM_PID
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "vLLM 서버가 정상 종료되었습니다"
        break
    else
        echo "vLLM 서버 오류 발생 (exit code: $EXIT_CODE)"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "재시도 전 정리 작업 수행..."
            cleanup_vllm_processes
            cleanup_pytorch_cache
            sleep 10
        fi
    fi
done

# 최종 정리
rm -f "$VLLM_PID_FILE"

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "vLLM 서버 시작 실패: 최대 재시도 횟수 초과"
    check_gpu_memory $MEMORY_THRESHOLD_MB
    exit 1
fi