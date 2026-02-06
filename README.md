# Music Looper

오디오 파일에서 완벽한 루프 포인트를 자동으로 찾아주는 데스크톱 앱입니다.

게임 BGM, 반복 재생 음악 등에서 자연스러운 루프 구간을 AI 기반으로 탐지하고, 다양한 형식으로 내보낼 수 있습니다.

![Music Looper Screenshot](docs/screenshot.png)

## 주요 기능

- **AI 루프 탐지** — Audio Spectrogram Transformer(AST) 기반 딥러닝 분석
- **비트 정렬** — madmom 기반 박자/마디 단위 루프 포인트 정렬
- **구조 분석** — allin1 기반 곡 구조(intro, verse, chorus 등) 인식 및 보정
- **이음새 미세조정** — 2단계 cross-correlation으로 샘플 단위 정밀 조정
- **파형 시각화** — wavesurfer.js 기반 파형 + 루프 구간 드래그 편집
- **갭리스 루프 재생** — 루프 구간 실시간 미리듣기
- **다양한 내보내기**
  - WAV 루프 구간 추출
  - OGG/WAV 루프 태그 (RPG Maker 호환 LOOPSTART/LOOPLENGTH)
  - 구간 분할 (intro/loop/outro)
  - 확장 루프 (N회 반복 + 크로스페이드)
  - 루프 정보 (JSON/TXT)

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 데스크톱 프레임워크 | Tauri 2.x (Rust) |
| 프론트엔드 | Next.js 16 + React 19 + shadcn/ui + wavesurfer.js |
| 백엔드 | Python (FastAPI + uvicorn) |
| 오디오 분석 | librosa, PyTorch, transformers (AST) |
| 비트/구조 향상 | madmom, allin1 |

### 아키텍처

```
Frontend (Next.js)
  ↕ fetch() / SSE
Python HTTP Server (FastAPI)
  ↕
core.py (분석 엔진)

Rust는 Python 프로세스 시작/종료 + 포트 전달만 담당
```

## 설치 및 빌드

### 요구사항

- [Node.js](https://nodejs.org/) 18+
- [pnpm](https://pnpm.io/)
- [Rust](https://rustup.rs/)
- [Python](https://www.python.org/) 3.10~3.12
- [uv](https://docs.astral.sh/uv/) (Python 패키지 매니저)

### 개발 모드

```bash
# 저장소 클론
git clone https://github.com/phj1081/music-looper-gui.git
cd music-looper-gui

# 백엔드 의존성 설치
cd backend
uv sync --all-extras --no-build-isolation-package madmom --no-build-isolation-package natten

# 프론트엔드 의존성 설치
cd ../frontend
pnpm install

# 개발 모드 실행
cd ..
pnpm dev:tauri
```

### 프로덕션 빌드

```bash
bash build.sh
```

PyInstaller로 Python 사이드카를 빌드한 뒤 Tauri가 `.app` 번들로 패키징합니다.

출력: `src-tauri/target/release/bundle/`

## 사용법

1. 앱을 실행하고 오디오 파일을 드래그 앤 드롭하거나 클릭하여 선택
2. 분석이 완료되면 발견된 루프 포인트 목록 확인
3. 각 루프 포인트를 클릭하여 미리듣기
4. 파형 위의 초록 구간을 드래그/리사이즈하여 수동 편집
5. 내보내기 메뉴에서 원하는 형식으로 저장

### 키보드 단축키

| 키 | 기능 |
|----|------|
| Space | 재생/일시정지 |
| L | 루프 토글 |
| E | 내보내기 메뉴 |
| 1-9 | 루프 포인트 선택 |

## 지원 형식

**입력**: MP3, WAV, FLAC, OGG, M4A

**출력**: WAV, OGG (루프 태그 포함), WAV with smpl chunk, JSON, TXT

## 라이선스

MIT License

## 크레딧

- [Audio Spectrogram Transformer (AST)](https://github.com/YuanGongND/ast) — 오디오 임베딩
- [madmom](https://github.com/CPJKU/madmom) — 비트/다운비트 탐지
- [allin1](https://github.com/mir-aidj/all-in-one) — 음악 구조 분석
- [librosa](https://librosa.org/) — 오디오 처리
- [wavesurfer.js](https://wavesurfer.xyz/) — 파형 시각화
