# NCP VM · Docker 배포 가이드 (popcorn-fastapi-server)

FastAPI + Docker Compose 기준입니다. 호스트(Ubuntu VM)에는 Python 가상환경 없이 **Docker만** 있으면 앱을 실행할 수 있습니다.

**처음 연습·구축은 `root`로 진행해도 됩니다.** 이 경우 프로젝트는 보통 `/root/popcorn-fastapi-server/` 에 두고, 아래 명령에서 `sudo` 는 생략해도 됩니다. 익숙해지면 [§1.6](#section-16-user-ssh)처럼 일반 사용자로 옮기면 됩니다.

---

## 1. VM 최초 기동 시 기본 설치

Ubuntu 22.04 LTS 권장. SSH 접속 후 **한 번** 실행합니다. **`root` 로 로그인 중이면 `sudo` 를 붙이지 않아도 됩니다.**

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

<a id="section-16-user-ssh"></a>

### 1.6 배포용 일반 사용자·SSH (나중에·권장)

**처음에는 이 절을 건너뛰고 `root` 로 [§5 체크리스트](#5-vm에서-처음부터-기동까지-순서-체크리스트)만 따라도 됩니다.**

`root`만으로도 배포는 됩니다. 다만 운영·보안을 위해 나중에 **전용 계정 + 홈 디렉터리**로 옮길 때 아래를 참고하면 됩니다. 사용자명은 예시로 `deploy` 를 씁니다 (원하면 `app`, `ubuntu` 등으로 바꿔도 됨).

**순서:** [§1.4 Docker 설치](#14-docker-engine--compose-공식-apt-저장소)까지 끝난 뒤, **root SSH 세션**에서 진행합니다.

#### 사용자 추가 (홈 자동 생성)

```bash
sudo adduser deploy
```

- 안내에 따라 비밀번호·이름 등 입력
- Ubuntu는 이때 **`/home/deploy/`** 가 만들어짐

패키지 설치 등 관리 작업을 이 계정에서 `sudo` 로 하려면:

```bash
sudo usermod -aG sudo deploy
```

#### Docker 그룹 (`sudo` 없이 `docker` / `docker compose`)

```bash
sudo usermod -aG docker deploy
```

#### SSH 공개키 로그인 설정

**방법 A — 지금 root로 키 로그인이 이미 될 때**  
같은 공개키를 `deploy` 에 복사합니다.

```bash
sudo mkdir -p /home/deploy/.ssh
sudo cp /root/.ssh/authorized_keys /home/deploy/.ssh/authorized_keys
sudo chown -R deploy:deploy /home/deploy/.ssh
sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
```

`/root/.ssh/authorized_keys` 가 없으면 방법 B.

**방법 B — 공개키를 직접 넣을 때**  
개발 PC에서 공개키 한 줄(예: `ssh-ed25519 AAAA...`)을 복사한 뒤, VM에서:

```bash
sudo mkdir -p /home/deploy/.ssh
sudo nano /home/deploy/.ssh/authorized_keys
# 붙여넣기 후 저장 (한 줄이면 됨)
sudo chown -R deploy:deploy /home/deploy/.ssh
sudo chmod 700 /home/deploy/.ssh
sudo chmod 600 /home/deploy/.ssh/authorized_keys
```

#### PuTTY로 `deploy` 접속

1. **Session** — Host Name: 서버 IP, Port: **22**
2. **Connection → SSH → Auth → Credentials** — **Private key file for authentication** 에 `.ppk` 지정
3. **Connection → Data → Auto-login username** 에 **`deploy`** 입력 (기존 `root` 대신)
4. **Open** 후 접속

비밀번호만 쓸 거면 2~3은 비우고, `adduser` 때 만든 비밀번호로 로그인하면 됩니다.

#### 접속 후 Docker 그룹 확인

`deploy` 로 다시 SSH 한 뒤:

```bash
docker run hello-world
```

`permission denied` 이면 **세션을 완전히 끊었다가** 다시 PuTTY로 접속하거나, 임시로 `newgrp docker` 후 재시도합니다.

#### (선택) 이후 앱 위치

```bash
cd ~
git clone <저장소-URL> popcorn-fastapi-server
```

#### (선택) root SSH 로그인 비활성화

`deploy` + 키 로그인이 안정적으로 동작하는 것을 확인한 뒤에만 검토합니다.  
`/etc/ssh/sshd_config` 에서 `PermitRootLogin no` 등 변경 → `sudo systemctl reload sshd`  
잘못 설정하면 접속이 막힐 수 있으니 콘솔(VNC/serial) 대비가 있을 때 진행합니다.

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

- 애플리케이션 코드: `fast.py`, `graph/` 패키지, `embedding_client.py` 등
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

`graph/` 패키지 / `embedding_client.py` 기준:

- **DB**: `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASS` 또는 `DB_PASSWORD`, `DB_PORT`
- **임베딩 API**: `EMBED_API_BASE_URL` (필수), 필요 시 `EMBED_API_PATH`, `EMBED_API_VERIFY_TLS`, `EMBED_API_TIMEOUT_SECONDS`, `EMBEDDING_DIMENSION`
- **LLM (선택)**: `LLM_MODEL_PATH`, `LLM_N_GPU_LAYERS`, `LLM_N_CTX` — 기본 Docker Compose는 CPU이며 `docker-compose.yml`에서 `LLM_N_GPU_LAYERS=0` 을 넣음

자세한 이름은 [`.env.example`](.env.example) 참고.

---

## 4. VM 접속 후 디렉터리 구조 (권장)

운영에서는 **root 대신 일반 사용자**(예: [§1.6](#section-16-user-ssh) 의 `deploy`)를 권장합니다. 아래는 **홈 디렉터리 기준** 예시입니다.

### 4.1 권장: 단일 앱 = 단일 Git 저장소

사용자 홈 아래에 프로젝트 하나를 둡니다.

```text
/home/deploy/                           # 또는 /home/<사용자명>/
└── popcorn-fastapi-server/             # git clone 한 루트 (= docker-compose.yml 위치)
    ├── Dockerfile
    ├── docker-compose.yml
    ├── requirements-docker.txt
    ├── fast.py
    ├── graph/
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
/home/deploy/
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

**`root` 로 진행하는 경우:** 2번은 건너뛰고, 4번은 예를 들어 `cd /root` 후 `git clone` 하여 `/root/popcorn-fastapi-server/` 를 씁니다.

1. [섹션 1](#1-vm-최초-기동-시-기본-설치) 패키지·Docker 설치  
2. (선택·나중에) [§1.6](#section-16-user-ssh) 일반 사용자·SSH·PuTTY — **처음엔 생략 가능**  
3. NCP ACG: SSH(22), 서비스 포트(8000 또는 80/443) 허용  
4. `git clone <저장소 URL>` → [섹션 4](#4-vm-접속-후-디렉터리-구조-권장) 구조로 배치  
5. `cp .env.example .env` 후 값 입력  
6. `models/` 에 GGUF 배치 또는 `bash scripts/download_model.sh`  
7. `docker compose up -d --build`  
8. `http://<공인IP>:8000/docs` 확인  

### 5.1 상세 설명 · 폴더 · `.env` 위치 (`root` 기준 예시)

아래에서 **프로젝트 루트**란 `docker-compose.yml` 과 같은 디렉터리입니다. `root` 로 클론했다면 보통 **`/root/popcorn-fastapi-server/`** 입니다.

| 단계 | 무엇을 하나 | 폴더 생성 | 비고 |
|------|------------|-----------|------|
| **1** | 패키지·Docker | (없음) | [§1](#1-vm-최초-기동-시-기본-설치) |
| **2** | 일반 사용자 | (선택) | 처음엔 생략 |
| **3** | ACG | (없음) | NCP 콘솔 |
| **4** | 코드 받기 | **`git clone` 이 폴더 전체를 만듦** | 별도 `mkdir` 불필요. 저장소 이름이 폴더명이 됨. |

```bash
cd /root
git clone https://github.com/<계정>/popcorn-fastapi-server.git popcorn-fastapi-server
cd /root/popcorn-fastapi-server
```

저장소 URL·로컬 폴더명(`popcorn-fastapi-server`)은 본인 환경에 맞게 바꿉니다. 이미 비어 있는 **부모 폴더**만 필요하면 `mkdir -p /root/work && cd /root/work` 후 그 안에서 `git clone` 하면 됩니다.

| **5** | 환경 변수 파일 | (없음) | **`.env` 는 반드시 프로젝트 루트**, 즉 `docker-compose.yml` 과 **같은 디렉터리**에 둡니다. `docker-compose.yml` 의 `env_file: .env` 가 이 경로를 가리킵니다. |

절대 경로 예:

```text
/root/popcorn-fastapi-server/.env
/root/popcorn-fastapi-server/.env.example   ← Git에서 옴
/root/popcorn-fastapi-server/docker-compose.yml
```

명령:

```bash
cd /root/popcorn-fastapi-server
cp .env.example .env
nano .env
```

| **6** | 모델 파일 | **`models/` 를 만들어 두는 것이 안전** | Compose 가 `./models:/app/models` 로 마운트합니다. 없으면 동작에 따라 자동 생성될 수 있으나, 권한·빈 디렉터리 문제를 피하려면 미리 만듭니다. |

```bash
cd /root/popcorn-fastapi-server
mkdir -p models
```

- **스크립트로 받기:** `bash scripts/download_model.sh` — 스크립트 안에서 `models` 를 `mkdir -p` 합니다.  
- **직접 올리기:** `scp` 등으로  
  `.../models/qwen2-1_5b-instruct-q4_k_m.gguf`  
  에 두고, 기본 `LLM_MODEL_PATH`(`/app/models/...`)와 파일 이름이 맞는지 확인합니다.

| **7** | 컨테이너 기동 | (없음) | **반드시 프로젝트 루트에서** 실행합니다. |

```bash
cd /root/popcorn-fastapi-server
docker compose up -d --build
```

| **8** | 브라우저 확인 | (없음) | ACG 에서 8000 등이 열려 있어야 함. |

**정리:** 만들어야 하는 디렉터리는 사실상 **`models/`** 뿐이고, `mkdir -p models` 한 번이면 됩니다. **`.env`** 는 **`/root/popcorn-fastapi-server/.env`** 처럼 **compose 파일 옆**에 둡니다.

코드만 갱신할 때:

```bash
# root 로 클론했다면 예: cd /root/popcorn-fastapi-server
cd ~/popcorn-fastapi-server
git pull origin main
docker compose up -d --build
```

`.env`와 `models/`는 그대로 둡니다.

---

## 6. 참고

- NCP 콘솔: VM 스펙은 Qwen2 1.5B Q4 CPU 추론 기준 **RAM 8GB 전후·디스크 여유** 권장.  
- 리버스 프록시(Nginx)·HTTPS는 이 문서 범위 밖이며, 운영 단계에서 ACG 80/443과 함께 구성하면 됩니다.
