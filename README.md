<div align="center">

# 🎵 Music Looper

**오디오 파일에서 완벽한 루프 포인트를 자동으로 찾아주는 데스크톱 앱**

게임 BGM, 반복 재생 음악 등에서 자연스러운 루프 구간을 AI 기반으로 탐지하고,
다양한 형식으로 내보낼 수 있습니다.

![Tauri](https://img.shields.io/badge/Tauri_2-24C8D8?style=flat-square&logo=tauri&logoColor=white)
![Next.js](https://img.shields.io/badge/Next.js_16-000000?style=flat-square&logo=next.js&logoColor=white)
![React](https://img.shields.io/badge/React_19-61DAFB?style=flat-square&logo=react&logoColor=black)
![Python](https://img.shields.io/badge/Python_3.10--3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![Rust](https://img.shields.io/badge/Rust-000000?style=flat-square&logo=rust&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

</div>

---

## 지원 OS

| OS | 기본 루프 탐지 | 비트 정렬 | 구조 분석 | 비고 |
|----|:-:|:-:|:-:|------|
| **macOS** (Apple Silicon) | O | O | O | |
| **macOS** (Intel) | O | O | O | CI 빌드 미제공 (로컬 빌드 가능) |
| **Linux** (x86_64) | O | O | O | |
| **Windows** (x86_64) | O | O | X | natten 미지원으로 구조 분석 불가 |

> **구조 분석**은 allin1 + natten 기반 기능입니다. natten이 Windows 공식 휠을 제공하지 않아 Windows에서는 해당 기능이 비활성화됩니다.

## 주요 기능

| 기능 | 설명 |
|------|------|
| **AI 루프 탐지** | Audio Spectrogram Transformer(AST) 기반 딥러닝 분석 |
| **비트 정렬** | madmom 기반 박자/마디 단위 루프 포인트 정렬 |
| **구조 분석** | allin1 기반 곡 구조(intro, verse, chorus 등) 인식 및 보정 |
| **이음새 미세조정** | 2단계 cross-correlation으로 샘플 단위 정밀 조정 |
| **파형 시각화** | wavesurfer.js 기반 파형 + 루프 구간 드래그 편집 |
| **갭리스 루프 재생** | 루프 구간 실시간 미리듣기 |
| **다양한 내보내기** | WAV/OGG 루프 태그, 구간 분할, 확장 루프, 루프 정보(JSON/TXT) |

## 아키텍처

```
┌─────────────────────────────────────────────────┐
│                   Tauri 2.x (Rust)              │
│  ┌─────────────┐         ┌────────────────────┐ │
│  │  sidecar.rs  │ spawn  │  Python Sidecar    │ │
│  │  commands.rs │───────▶│  (FastAPI+uvicorn)  │ │
│  └──────┬──────┘ port    └────────┬───────────┘ │
│         │ get_server_port         │             │
│  ┌──────┴──────────────┐  ┌──────┴───────────┐ │
│  │  Frontend (Next.js)  │  │  core.py         │ │
│  │  React 19 + shadcn   │──│  deep_analyzer   │ │
│  │  wavesurfer.js       │  │  allin1_enhancer │ │
│  └─────────────────────┘  └──────────────────┘ │
│     fetch() / SSE / dialog                      │
└─────────────────────────────────────────────────┘
```

- **Rust** — Python 프로세스 생명주기 관리 + 포트 전달만 담당
- **Frontend** → `fetch(http://127.0.0.1:PORT/...)` → Python FastAPI → core.py
- **Progress** — SSE(`GET /progress`) 기반 실시간 진행률
- **Audio** — `GET /audio` → WAV FileResponse (wavesurfer.js가 직접 로드)

## 프로젝트 구조

```
music-looper-gui/
├── frontend/                  # Next.js 16 (정적 빌드)
│   └── src/
│       ├── app/               # 페이지 (page.tsx, layout.tsx)
│       ├── components/        # UI 컴포넌트
│       │   ├── ui/            # shadcn/ui 기본 컴포넌트
│       │   ├── waveform.tsx   # wavesurfer.js 파형 시각화
│       │   ├── loop-table.tsx # 루프 포인트 목록
│       │   ├── loop-editor.tsx# 루프 구간 수동 편집
│       │   ├── player-controls.tsx
│       │   └── export-menu.tsx
│       └── lib/
│           ├── api.ts         # fetch/SSE + Tauri dialog 브릿지
│           ├── looping-webaudio-player.ts
│           └── utils.ts
├── backend/                   # Python (FastAPI 사이드카)
│   ├── http_server.py         # FastAPI + uvicorn (진입점)
│   ├── core.py                # 분석 엔진 (headless)
│   ├── deep_analyzer.py       # AST 기반 루프 탐지
│   ├── allin1_enhancer.py     # 구조 분석 보정
│   └── pyproject.toml         # uv 패키지 정의
├── src-tauri/                 # Tauri 2.x (Rust)
│   ├── src/
│   │   ├── lib.rs             # Tauri 플러그인 등록
│   │   ├── main.rs            # 진입점
│   │   ├── sidecar.rs         # Python 프로세스 관리
│   │   └── commands.rs        # Tauri 커맨드 (get_server_port)
│   ├── tauri.conf.json
│   └── binaries/              # PyInstaller 사이드카 바이너리
├── build.sh                   # 프로덕션 빌드 스크립트
└── package.json               # 루트 스크립트 (dev, build, lint)
```

## 기술 스택

| 레이어 | 기술 |
|--------|------|
| 데스크톱 프레임워크 | Tauri 2.x (Rust) |
| 프론트엔드 | Next.js 16 + React 19 + shadcn/ui + wavesurfer.js |
| 백엔드 | Python 3.10~3.12 (FastAPI + uvicorn) |
| 오디오 분석 | librosa, PyTorch, transformers (AST) |
| 비트/구조 보정 | madmom, allin1 |
| 패키징 | PyInstaller (사이드카) + Tauri (앱 번들) |

## 설치 및 빌드

### 요구사항

- [Tauri 시스템 의존성](https://v2.tauri.app/start/prerequisites/) (OS별 안내 참고)
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

# 초기 설정 (의존성 설치 + dev 사이드카 생성, 최초 1회)
pnpm setup

# 개발 모드 실행
pnpm dev
```

> `pnpm setup`은 백엔드/프론트엔드 의존성 설치를 수행합니다 (최초 1회).
> `pnpm dev` 실행 시 dev 사이드카 래퍼가 자동으로 세팅되므로, 프로덕션 빌드 후에도 별도 설정 없이 바로 개발 모드로 전환할 수 있습니다.
> Python 코드 수정 시 앱만 재시작하면 반영됩니다 (사이드카 재빌드 불필요).

> **Windows 사용자**: Git Bash 또는 WSL에서 실행해주세요.

### 프로덕션 빌드

```bash
pnpm build
```

> **Windows 사용자**: Git Bash 또는 WSL에서 실행해주세요.

PyInstaller로 Python 사이드카를 빌드한 뒤 Tauri가 앱 번들로 패키징합니다.

| OS | 출력 형식 |
|----|-----------|
| macOS | `.app` / `.dmg` |
| Linux | `.AppImage` / `.deb` |
| Windows | `.msi` / `.exe` |

출력: `src-tauri/target/release/bundle/`

## 설치

[Releases](https://github.com/phj1081/music-looper-gui/releases) 페이지에서 OS에 맞는 설치 파일을 다운로드하세요.

> **macOS 사용자**: 앱이 코드 서명되어 있지 않아 처음 실행 시 보안 경고가 나타날 수 있습니다.
>
> 1. `.dmg`를 열고 앱을 Applications 폴더로 드래그
> 2. 앱을 **우클릭 → 열기** → 확인 대화상자에서 **열기** 클릭
>
> 최초 1회만 필요하며, 이후에는 정상적으로 실행됩니다.

## 사용법

1. 앱을 실행하고 오디오 파일을 **드래그 앤 드롭**하거나 클릭하여 선택
2. 분석이 완료되면 발견된 루프 포인트 목록 확인
3. 각 루프 포인트를 클릭하여 미리듣기
4. 파형 위의 초록 구간을 드래그/리사이즈하여 수동 편집
5. 내보내기 메뉴에서 원하는 형식으로 저장

### 키보드 단축키

| 키 | 기능 |
|----|------|
| `Space` | 재생/일시정지 |
| `L` | 루프 토글 |
| `E` | 내보내기 메뉴 |
| `1`~`9` | 루프 포인트 선택 |

## 지원 형식

| 구분 | 형식 |
|------|------|
| **입력** | MP3, WAV, FLAC, OGG, M4A |
| **출력** | WAV, OGG (루프 태그), WAV (smpl chunk), JSON, TXT |

### 내보내기 옵션

- **루프 구간 추출** — 선택한 구간만 WAV로 저장
- **루프 태그** — OGG/WAV에 LOOPSTART/LOOPLENGTH 메타데이터 삽입 (RPG Maker 호환)
- **구간 분할** — intro / loop / outro 개별 파일로 분리
- **확장 루프** — N회 반복 + 크로스페이드 적용
- **루프 정보** — JSON/TXT로 루프 포인트 좌표 내보내기

## 라이선스

MIT License

## 크레딧

- [Audio Spectrogram Transformer (AST)](https://github.com/YuanGongND/ast) — 오디오 임베딩
- [madmom](https://github.com/CPJKU/madmom) — 비트/다운비트 탐지
- [allin1](https://github.com/mir-aidj/all-in-one) — 음악 구조 분석
- [librosa](https://librosa.org/) — 오디오 처리
- [wavesurfer.js](https://wavesurfer.xyz/) — 파형 시각화
- [Tauri](https://tauri.app/) — 데스크톱 앱 프레임워크
