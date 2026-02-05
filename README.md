# Music Looper

오디오 파일에서 완벽한 루프 포인트를 자동으로 찾아주는 데스크톱 앱입니다.

![Music Looper Screenshot](docs/screenshot.png)

## 주요 기능

### 루프 포인트 자동 탐지
- **Audio Spectrogram Transformer (AST)** 기반 딥러닝 분석
- 임베딩 기반 **recurrence matrix + path enhancement**로 반복(구조) 구간 후보 탐색
- 원본 샘플레이트에서 이음새(seam) **상관/제로크로싱 기반 미세조정**
- 여러 후보 중 최적의 루프 포인트 순위 제공
- (선택) madmom 설치 시 마디(다운비트) 정렬, allin1 설치 시 구조(섹션) 기반 점수 보정

### 게임 BGM용 내보내기
게임 개발에 필요한 다양한 내보내기 옵션 지원:

| 기능 | 설명 |
|------|------|
| **루프 태그 포함 (OGG)** | `LOOPSTART`, `LOOPLENGTH` 태그 삽입 (RPG Maker 등 호환) |
| **루프 태그 포함 (WAV)** | `smpl` 청크에 루프 포인트 기록 (샘플러/게임 엔진 호환) |
| **섹션별 분리** | 인트로/루프/아웃트로를 개별 파일로 저장 |
| **확장 버전** | 인트로 + (루프 × N회) + 아웃트로 구조로 긴 버전 생성 |
| **루프 정보** | 샘플 번호, 시간 정보를 JSON/TXT로 저장 |

## 설치

### 요구사항
- Python 3.10+
- Node.js 18+
- [uv](https://github.com/astral-sh/uv) (Python 패키지 매니저)

### 설치 방법

```bash
# 저장소 클론
git clone https://github.com/your-username/music-looper-gui.git
cd music-looper-gui

# 프론트엔드
cd frontend && pnpm install

# 백엔드
cd ../backend && uv sync
```

#### 선택 기능(권장)
더 나은 결과를 원하면 optional dependency를 함께 설치하세요.

```bash
cd backend
# madmom(비트/마디 정렬) + allin1(구조 분석)
uv sync --all-extras --no-build-isolation-package madmom --no-build-isolation-package natten
# 또는 필요한 것만
uv sync --extra beat --no-build-isolation-package madmom
uv sync --extra structure --no-build-isolation-package natten --no-build-isolation-package madmom
```

macOS에서 `madmom`/`natten` 빌드가 실패하면(빌드 툴/의존성 이슈), 아래를 먼저 실행한 뒤 다시 시도하세요:

```bash
cd backend
uv sync
uv pip install --python .venv/bin/python cython cmake ninja
```

실행 시 기능을 강제로 끄고 싶으면(성능/검증용):

```bash
cd backend
MUSIC_LOOPER_DISABLE_BEAT_ALIGNMENT=1 uv run python app.py
MUSIC_LOOPER_DISABLE_STRUCTURE=1 uv run python app.py
```

## 실행

```bash
# 빌드(프론트엔드 export + backend/static 복사)
pnpm build

# 앱 실행
pnpm start
```

### 개발 모드

```bash
# 프론트엔드 dev 서버 + 데스크톱 앱 동시 실행
pnpm dev
```

## 사용법

1. 앱을 실행하고 오디오 파일을 드래그 앤 드롭하거나 클릭하여 선택
2. 분석이 완료되면 발견된 루프 포인트 목록 확인
3. 각 루프 포인트를 클릭하여 미리듣기
4. 원하는 루프를 선택하고 내보내기 메뉴에서 원하는 형식으로 저장

## 지원 형식

### 입력
- MP3, WAV, FLAC, OGG, M4A

### 출력
- WAV (기본)
- OGG Vorbis (루프 태그 포함)
- WAV with smpl chunk (루프 태그 포함)
- JSON/TXT (루프 정보)

## 기술 스택

- **Backend**: Python, PyWebView, librosa, PyTorch, Transformers
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS, shadcn/ui
- **AI Model**: Audio Spectrogram Transformer (AST)

## 라이선스

MIT License
