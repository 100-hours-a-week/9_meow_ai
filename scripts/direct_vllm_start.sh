#!/bin/bash

# vLLM 서버 직접 시작 스크립트
# 배포 환경에서 안정적인 실행을 위한 백업 방안

echo "=== vLLM 직접 시작 스크립트 ==="
echo "모델: ${VLLM_MODEL_PATH:-haebo/meow-clovax-v2}"
echo "포트: ${VLLM_PORT:-8001}"

# 메모리 임계값 설정
MEMORY_THRESHOLD_MB=15000  # 15GB
MAX_RETRIES=3
RETRY_COUNT=0

# PID 파일 경로
VLLM_PID_FILE="/tmp/vllm_server.pid"

# 기존 vLLM 프로세스 정리 (안전한 방법)
cleanup_vllm_processes() {
    echo "=== vLLM 프로세스 정리 ==="
    
    # PID 파일 기반 정리
    if [ -f "$VLLM_PID_FILE" ]; then
        OLD_PID=$(cat "$VLLM_PID_FILE")
        if kill -0 "$OLD_PID" 2>/dev/null; then
            echo "기존 vLLM 프로세스 종료 중 (PID: $OLD_PID)"
            kill -TERM "$OLD_PID"
            sleep 5
            kill -KILL "$OLD_PID" 2>/dev/null || true
        fi
        rm -f "$VLLM_PID_FILE"
    fi
    
    # 프로세스명 기반 정리 (더 안전한 방법)
    pkill -f "vllm.entrypoints.openai.api_server" || true
    pkill -f "python.*vllm.*api_server" || true
    sleep 3
}

# GPU 메모리 상태 체크
check_gpu_memory() {
    local used_memory
    used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    echo "현재 GPU 메모리 사용량: ${used_memory}MB"
    
    if [ "$used_memory" -gt "$MEMORY_THRESHOLD_MB" ]; then
        echo "경고: GPU 메모리 사용량이 임계값을 초과했습니다 (${used_memory}MB > ${MEMORY_THRESHOLD_MB}MB)"
        return 1
    fi
    return 0
}

# PyTorch 캐시 정리
cleanup_pytorch_cache() {
    echo "=== PyTorch 캐시 정리 ==="
    python3 -c "
import torch
import gc
try:
    if torch.cuda.is_available():
        print(f'CUDA 디바이스 수: {torch.cuda.device_count()}')
        for i in range(torch.cuda.device_count()):
            torch.cuda.set_device(i)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            print(f'GPU {i} 캐시 정리 완료')
        # 추가 메모리 정리
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('CUDA 사용 불가')
except Exception as e:
    print(f'캐시 정리 중 오류: {e}')
" || true
}

# 시작 전 정리 및 체크
echo "=== 시작 전 시스템 체크 ==="

# 1. 기존 프로세스 정리
cleanup_vllm_processes

# 2. PyTorch 캐시 정리
cleanup_pytorch_cache

# 3. GPU 메모리 상태 체크
if ! check_gpu_memory; then
    echo "GPU 메모리 상태가 좋지 않습니다. 추가 정리 수행..."
    # 추가 정리 로직
    cleanup_pytorch_cache
    sleep 5
    if ! check_gpu_memory; then
        echo "GPU 메모리 정리 실패. 종료합니다."
        exit 1
    fi
fi

# 환경변수 출력
echo "=== vLLM 설정 정보 ==="
echo "GPU 메모리 사용률: ${VLLM_GPU_MEMORY_UTILIZATION:-0.4}"
echo "최대 모델 길이: ${VLLM_MAX_MODEL_LEN:-1024}"
echo "동시 시퀀스 수: ${VLLM_MAX_NUM_SEQS:-6}"
echo "청크 크기: ${VLLM_CHUNK_SIZE:-512}"
echo "청크 프리필 활성화: ${VLLM_ENABLE_CHUNKED_PREFILL:-true}"
echo "프리픽스 캐싱: ${VLLM_ENABLE_PREFIX_CACHING:-false}"

# vLLM 서버 시작 루프
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "vLLM 서버 시작 시도 ($((RETRY_COUNT + 1))/$MAX_RETRIES)..."
    
    # 시작 전 메모리 체크
    if ! check_gpu_memory; then
        echo "시작 전 메모리 체크 실패"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        continue
    fi
    
    # vLLM 서버 시작
    python -m vllm.entrypoints.openai.api_server \
        --model "${VLLM_MODEL_PATH:-haebo/meow-clovax-v2}" \
        --host "${VLLM_HOST:-0.0.0.0}" \
        --port "${VLLM_PORT:-8001}" \
        --served-model-name "${VLLM_SERVED_MODEL_NAME:-meow-clovax-v2}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-1024}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-6}" \
        --enable-chunked-prefill \
        --chunk-size "${VLLM_CHUNK_SIZE:-512}" \
        --disable-log-requests \
        --trust-remote-code &
    
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
    check_gpu_memory
    exit 1
fi