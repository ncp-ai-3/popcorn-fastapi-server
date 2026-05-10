#!/usr/bin/env bash
# PuTTY/SSH로 Linux VM에 접속한 뒤, 저장소 루트에서 실행하거나
#   bash scripts/download_model.sh
# 직접 URL을 넘기려면:
#   MODEL_URL='https://...' bash scripts/download_model.sh
# Hugging Face 게이트 모델이면:
#   HF_TOKEN=hf_xxx MODEL_URL='https://...' bash scripts/download_model.sh

set -euo pipefail

# 저장소 루트( docker-compose.yml 이 있는 디렉터리 )
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODEL_DIR="${REPO_ROOT}/models"
MODEL_FILE="${MODEL_FILE:-qwen2-1_5b-instruct-q4_k_m.gguf}"
DEST="${MODEL_DIR}/${MODEL_FILE}"

# 기본값: Qwen2 1.5B Instruct Q4_K_M (공개). 다른 파일을 쓰면 MODEL_URL 로 덮어쓰기.
DEFAULT_URL="https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1_5b-instruct-q4_k_m.gguf"

MODEL_URL="${MODEL_URL:-${1:-$DEFAULT_URL}}"

mkdir -p "${MODEL_DIR}"

if [[ -f "${DEST}" ]]; then
  echo "이미 존재: ${DEST}"
  echo "다시 받으려면 삭제 후 재실행하세요."
  exit 0
fi

echo "다운로드 대상: ${DEST}"
echo "URL: ${MODEL_URL}"

if command -v curl >/dev/null 2>&1; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    curl -fL --retry 3 --retry-delay 5 \
      -H "Authorization: Bearer ${HF_TOKEN}" \
      -o "${DEST}.part" "${MODEL_URL}"
  else
    curl -fL --retry 3 --retry-delay 5 -o "${DEST}.part" "${MODEL_URL}"
  fi
elif command -v wget >/dev/null 2>&1; then
  if [[ -n "${HF_TOKEN:-}" ]]; then
    wget -O "${DEST}.part" --header="Authorization: Bearer ${HF_TOKEN}" "${MODEL_URL}"
  else
    wget -O "${DEST}.part" "${MODEL_URL}"
  fi
else
  echo "curl 또는 wget 이 필요합니다. sudo apt-get install -y curl"
  exit 1
fi

mv "${DEST}.part" "${DEST}"
echo "완료: ${DEST}"
ls -lh "${DEST}"
