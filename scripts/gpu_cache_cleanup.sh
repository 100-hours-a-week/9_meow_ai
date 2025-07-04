#!/bin/bash

# GPU 캐시 정리 및 vLLM 관리 유틸리티 스크립트
# 다른 스크립트에서 source하여 사용 가능

# vLLM 프로세스 정리 함수
cleanup_vllm_processes() {
    echo "=== vLLM 프로세스 정리 ==="
    
    # PID 파일 기반 정리
    local pid_file="/tmp/vllm_server.pid"
    if [ -f "$pid_file" ]; then
        local old_pid=$(cat "$pid_file")
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "기존 vLLM 프로세스 종료 중 (PID: $old_pid)"
            kill -TERM "$old_pid"
            sleep 5
            kill -KILL "$old_pid" 2>/dev/null || true
        fi
        rm -f "$pid_file"
    fi
    
    # 프로세스명 기반 정리
    echo "vLLM 관련 프로세스 정리 중..."
    pkill -f "vllm.entrypoints.openai.api_server" || true
    pkill -f "python.*vllm.*api_server" || true
    sleep 3
}

# PyTorch 캐시 정리 함수
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
        # 강제 가비지 컬렉션
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print('CUDA 사용 불가')
except Exception as e:
    print(f'캐시 정리 중 오류: {e}')
" || true
}

# GPU 메모리 상태 확인 함수
check_gpu_memory() {
    local threshold_mb=${1:-15000}  # 기본값 15GB
    local used_memory
    
    echo "=== GPU 메모리 상태 확인 ==="
    used_memory=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null || echo "0")
    
    echo "현재 GPU 메모리 사용량: ${used_memory}MB"
    echo "임계값: ${threshold_mb}MB"
    
    if [ "$used_memory" -gt "$threshold_mb" ]; then
        echo "경고: GPU 메모리 사용량이 임계값을 초과했습니다"
        return 1
    fi
    return 0
}

# GPU 메모리 상태 출력 함수
show_gpu_memory() {
    echo "GPU 메모리 상태:"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader || true
}

# 전체 정리 함수
full_cleanup() {
    echo "=== 전체 GPU 캐시 정리 시작 ==="
    
    show_gpu_memory
    cleanup_vllm_processes
    cleanup_pytorch_cache
    show_gpu_memory
    
    echo "=== 전체 GPU 캐시 정리 완료 ==="
}

# 스크립트가 직접 실행된 경우 전체 정리 수행
if [ "${BASH_SOURCE[0]}" == "${0}" ]; then
    full_cleanup
fi 