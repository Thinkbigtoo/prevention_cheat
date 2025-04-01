# Gaze Tracking System

시선 추적 시스템은 OpenCV와 MediaPipe를 사용하여 사용자의 시선을 추적하고, 손 제스처를 인식하는 시스템입니다.

## 기능

- 실시간 시선 추적
- 손 제스처 인식
- MariaDB를 통한 결과 데이터 저장
- Docker를 통한 컨테이너화

## 시스템 요구사항

- Python 3.9 이상
- OpenCV
- MediaPipe
- MariaDB
- Docker

## 설치 방법

1. 저장소 클론
```bash
git clone https://github.com/your-username/gaze-tracking.git
cd gaze-tracking
```

2. 필요한 패키지 설치
```bash
pip install -r requirements.txt
```

3. Docker 이미지 빌드 및 실행
```bash
./build_and_run.sh
```

## 사용 방법

1. 프로그램 실행
```bash
python example.py
```

2. 종료
- ESC 키를 눌러 프로그램 종료

## Docker 지원

### 빌드
```bash
docker build -t gaze-tracking .
```

### 실행
```bash
docker run --device=/dev/video0:/dev/video0 gaze-tracking
```

## 라즈베리파이 지원

라즈베리파이에서 실행하기 전에 다음 사항을 확인하세요:

1. 카메라 모듈 활성화
2. 필요한 시스템 패키지 설치
3. Docker 설치

## 라이선스

이 프로젝트는 MIT 라이선스를 따릅니다.

라즈베리파이에서 실행시 
docker run --device=/dev/video0:/dev/video0 gaze-tracking

도커 네트워크 사용시
docker network create gaze-network
docker run --network gaze-network --device=/dev/video0:/dev/video0 gaze-tracking

GUI가 필요한 경우 X11 서버가 설치되어 있어야 함

권한 설정
# 스크립트 실행 권한 부여
chmod +x build_and_run.sh

/etc/modules에 다음이 있는지 확인해야함
bcm2835-v4l2