#!/bin/bash

# vLLM 서버 직접 시작 스크립트
# 배포 환경에서 안정적인 실행을 위한 백업 방안

echo "=== vLLM 직접 시작 스크립트 ==="
echo "모델: ${VLLM_MODEL_PATH:-haebo/meow-clovax-v2}"
echo "포트: ${VLLM_PORT:-8001}"

# 환경 변수 설정
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}

# 최대 재시도 횟수
MAX_RETRIES=3
RETRY_COUNT=0

while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    echo "vLLM 서버 시작 시도 ($((RETRY_COUNT + 1))/$MAX_RETRIES)..."
    
    python -m vllm.entrypoints.openai.api_server \
        --model "${VLLM_MODEL_PATH:-haebo/meow-clovax-v2}" \
        --host "${VLLM_HOST:-0.0.0.0}" \
        --port "${VLLM_PORT:-8001}" \
        --served-model-name "${VLLM_SERVED_MODEL_NAME:-meow-clovax-v2}" \
        --gpu-memory-utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.6}" \
        --max-model-len "${VLLM_MAX_MODEL_LEN:-1536}" \
        --max-num-seqs "${VLLM_MAX_NUM_SEQS:-12}" \
        --disable-log-requests \
        --trust-remote-code
    
    EXIT_CODE=$?
    
    if [ $EXIT_CODE -eq 0 ]; then
        echo "vLLM 서버가 정상 종료되었습니다"
        break
    else
        echo "vLLM 서버 오류 발생 (exit code: $EXIT_CODE)"
        RETRY_COUNT=$((RETRY_COUNT + 1))
        
        if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
            echo "10초 후 재시도합니다..."
            sleep 10
        fi
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo "vLLM 서버 시작 실패: 최대 재시도 횟수 초과"
    exit 1
fi 