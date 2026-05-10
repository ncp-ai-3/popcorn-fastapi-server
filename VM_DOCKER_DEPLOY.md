# NCP VM · Docker 배포 가이드 (popcorn-fastapi-server)

FastAPI + Docker Compose 기준입니다. 호스트(Ubuntu VM)에는 Python 가상환경 없이 **Docker만** 있으면 앱을 실행할 수 있습니다.

---

## 1. VM 최초 기동 시 기본 설치

Ubuntu 22.04 LTS 권장. SSH 접속 후 **한 번** 실행합니다.

### 1.1 시스템 업데이트

```bash
sudo apt update && sudo apt upgrade -y
```

### 1.2 코드·다운로드·운영 도구

```bash
sudo apt install -y \
  git \
  curl \
  wget \
  vim \
  nano \
  htop \
  tmux \
  unzip
```

### 1.3 (선택) DB 연결 테스트용

VM에서 `psql`로 PostgreSQL만 확인할 때:

```bash
sudo apt install -y postgresql-client
```

### 1.4 Docker Engine + Compose (공식 apt 저장소)

[Install Docker Engine on Ubuntu](https://docs.docker.com/engine/install/ubuntu/) 의 **Install using the apt repository** 절과 동일한 흐름입니다.

#### (선택) 충돌할 수 있는 비공식 패키지 제거

공식 문서의 *Uninstall old versions* 를 참고합니다. 아래는 한 번에 제거 시도하는 예시입니다.

```bash
sudo apt remove -y docker.io docker-compose docker-compose-v2 docker-doc podman-docker containerd runc
```

해당 패키지가 없으면 `apt` 가 알려 주고 넘어가면 됩니다.

#### Docker apt 저장소 등록

```bash
sudo apt update
sudo apt install -y ca-certificates curl
sudo install -m 0755 -d /etc/apt/keyrings
sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
sudo chmod a+r /etc/apt/keyrings/docker.asc

sudo tee /etc/apt/sources.list.d/docker.sources <<EOF
Types: deb
URIs: https://download.docker.com/linux/ubuntu
Suites: $(. /etc/os-release && echo "${UBUNTU_CODENAME:-$VERSION_CODENAME}")
Components: stable
Architectures: $(dpkg --print-architecture)
Signed-By: /etc/apt/keyrings/docker.asc
EOF

sudo apt update
```

#### 패키지 설치

```bash
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

#### 서비스 및 설치 확인

```bash
sudo systemctl status docker
# Active 가 아니면: sudo systemctl start docker

sudo docker run hello-world
sudo docker compose version
docker --version
```

#### (권장) `sudo` 없이 Docker 사용

[Linux post-install steps](https://docs.docker.com/engine/install/linux-postinstall/) — 관리 사용자를 `docker` 그룹에 추가합니다.

```bash
sudo groupadd docker 2>/dev/null || true
sudo usermod -aG docker "$USER"
```

**SSH 세션을 끊고 다시 접속**한 뒤, `sudo` 없이 `docker run hello-world` 가 되는지 확인합니다.

**Docker 기준 배포**에서는 호스트에 `python3-venv`, `pip install fastapi` 등은 **필수 아님** (컨테이너에 포함됨).

### 1.5 (선택) GPU VM을 쓸 때만

현재 저장소의 기본 Docker 이미지는 **CPU용 `llama-cpp-python`** 입니다. GPU 추론으로 바꿀 계획이 있을 때만 호스트에 NVIDIA 드라이버·CUDA·`nvidia-container-toolkit` 등을 별도로 맞춥니다.

---

## 2. Dockerfile은 어디서 만들고, VM에서는 어떻게 쓰나

| 단계 | 설명 |
|------|------|
| **개발 PC** | 저장소 루트에 이미 있는 `Dockerfile`, `docker-compose.yml`, `requirements-docker.txt` 를 유지·수정합니다. |
| **Git** | `main` 등에 **push** 합니다. (`Dockerfile`은 Git에 포함됩니다.) |
| **VM** | `git clone` 또는 `git pull`로 **같은 파일을 받습니다.** VM에서 Dockerfile을 “처음부터 새로 작성”할 필요는 없습니다. |
| **VM** | 프로젝트 루트에서 `docker compose build` / `docker compose up -d --build` 로 이미지를 **빌드**합니다. |

로컬에서 빌드 검증 예시 (개발 PC):

```bash
docker build -t popcorn-api:local .
```

VM에서 기동:

```bash
cd /path/to/popcorn-fastapi-server
docker compose up -d --build
```

---

## 3. Git에 올리는 것 vs VM에만 두는 것

### 3.1 Git에 포함 (저장소에 push)

- 애플리케이션 코드: `fast.py`, `graph.py`, `embedding_client.py` 등
- 컨테이너 정의: `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `requirements-docker.txt`
- 예시 환경변수: `.env.example` (비밀 값 없음)
- 보조 스크립트: `scripts/download_model.sh` 등
- 본 가이드: `VM_DOCKER_DEPLOY.md`

### 3.2 VM에만 두고 Git에는 넣지 않음

| 파일/디렉터리 | 내용 |
|---------------|------|
| **`.env`** | `DB_*`, `EMBED_API_BASE_URL` 등 실제 비밀·URL. VM에서 `cp .env.example .env` 후 채움. |
| **`models/*.gguf`** | LLM 가중치. 용량이 크고 `.gitignore` 대상. `scripts/download_model.sh` 또는 `scp`로 VM에 배치. |
| **(실수 방지)** | 로컬 개발용 `.env`를 그대로 `scp`하지 말 것. VM용 값만 넣을 것. |

`.gitignore`에 이미 있는 항목(예: `.env`, `*.gguf`)은 커밋하지 않습니다.

### 3.3 `.env`에 넣을 변수 (요약)

`graph.py` / `embedding_client.py` 기준:

- **DB**: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS` 또는 `DB_PASSWORD`, `DB_PORT`
- **임베딩 API**: `EMBED_API_BASE_URL` (필수), 필요 시 `EMBED_API_PATH`, `EMBED_API_VERIFY_TLS`, `EMBED_API_TIMEOUT_SECONDS`, `EMBEDDING_DIMENSION`
- **LLM (선택)**: `LLM_MODEL_PATH`, `LLM_N_GPU_LAYERS`, `LLM_N_CTX` — 기본 Docker Compose는 CPU이며 `docker-compose.yml`에서 `LLM_N_GPU_LAYERS=0` 을 넣음

자세한 이름은 [`.env.example`](.env.example) 참고.

---

## 4. VM 접속 후 디렉터리 구조 (권장)

운영에서는 **root 계정 대신 일반 사용자**(예: `ubuntu`)로 두는 것을 권장합니다. 아래는 **홈 디렉터리 기준** 예시입니다.

### 4.1 권장: 단일 앱 = 단일 Git 저장소

사용자 홈 아래에 프로젝트 하나를 둡니다.

```text
/home/ubuntu/                          # 또는 /home/<사용자명>/
└── popcorn-fastapi-server/             # git clone 한 루트 (= docker-compose.yml 위치)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements-docker.txt
    ├── fast.py
    ├── graph.py
    ├── embedding_client.py
    ├── .env                            # VM에서만 생성·편집 (Git 제외)
    ├── .env.example                    # Git에서 옴
    ├── models/                         # VM에서만 채움 (Git 제외)
    │   └── qwen2-1_5b-instruct-q4_k_m.gguf
    └── scripts/
        └── download_model.sh
```

- **`docker compose`** 는 반드시 **`docker-compose.yml` 이 있는 디렉터리** (`popcorn-fastapi-server/`)에서 실행합니다.
- **볼륨** `./models` 는 이 루트 기준 `./models` → 컨테이너 `/app/models` 입니다.

### 4.2 여러 서비스를 나누고 싶을 때 (선택)

같은 사용자 홈 아래에 앱별 폴더만 늘립니다.

```text
/home/ubuntu/
├── popcorn-fastapi-server/     # 본 저장소
└── (추후) other-service/
```

### 4.3 root 계정 홈을 쓰는 경우 (비권장·예시만)

PuTTY로 `root`만 쓰는 환경이라면 구조는 동일하고 경로만 바뀝니다.

```text
/root/popcorn-fastapi-server/
├── docker-compose.yml
├── .env
├── models/
│   └── ...
└── ...
```

보안·감사 측면에서 가능하면 **일반 사용자 + sudo** 를 권장합니다.

---

## 5. VM에서 처음부터 기동까지 순서 (체크리스트)

1. [섹션 1](#1-vm-최초-기동-시-기본-설치) 패키지·Docker 설치  
2. NCP ACG: SSH(22), 서비스 포트(8000 또는 80/443) 허용  
3. `git clone <저장소 URL>` → [섹션 4](#4-vm-접속-후-디렉터리-구조-권장) 구조로 배치  
4. `cp .env.example .env` 후 값 입력  
5. `models/` 에 GGUF 배치 또는 `bash scripts/download_model.sh`  
6. `docker compose up -d --build`  
7. `http://<공인IP>:8000/docs` 확인  

코드만 갱신할 때:

```bash
cd ~/popcorn-fastapi-server
git pull origin main
docker compose up -d --build
```

`.env`와 `models/`는 그대로 둡니다.

---

## 6. 참고

- NCP 콘솔: VM 스펙은 Qwen2 1.5B Q4 CPU 추론 기준 **RAM 8GB 전후·디스크 여유** 권장.  
- 리버스 프록시(Nginx)·HTTPS는 이 문서 범위 밖이며, 운영 단계에서 ACG 80/443과 함께 구성하면 됩니다.
