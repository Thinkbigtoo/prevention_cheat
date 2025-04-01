FROM ubuntu:18.04

# 시스템 패키지 설치
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    cmake \
    libsm6 \
    libxext6 \
    libxrender1 \
    libfontconfig1 \
    libglib2.0-0 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Python 패키지 업그레이드
RUN pip3 install --upgrade pip

# 작업 디렉토리 설정
WORKDIR /home/GazeTracking

# 애플리케이션 코드 복사
COPY . .

# 필요한 Python 패키지 설치
RUN pip3 install mediapipe
RUN pip3 install -r requirements.txt

# 카메라 접근을 위한 장치 마운트
VOLUME /dev/video0:/dev/video0

# 환경 변수 설정
ENV PYTHONUNBUFFERED=1
ENV DISPLAY=:0

# 실행 명령
CMD ["python3", "example.py"]
