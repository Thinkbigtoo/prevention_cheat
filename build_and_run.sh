#!/bin/bash

IMAGE_NAME=gaze-tracking

# 라즈베리파이 환경 확인
if [ -f /etc/rpi-issue ]; then
    echo "라즈베리파이 환경 감지"
    # 라즈베리파이 카메라 모듈 확인
    if [ -e /dev/video0 ]; then
        echo "카메라 모듈이 감지되었습니다."
    else
        echo "경고: 카메라 모듈이 감지되지 않았습니다."
        exit 1
    fi
fi

# X11 디스플레이 서버 설정 (GUI가 필요한 경우)
if [ -n "$DISPLAY" ]; then
    xhost local:root
fi

# 도커 이미지 빌드
if [ "$(docker images -q ${IMAGE_NAME})" == "" ]; then
    echo "도커 이미지 빌드 중..."
    docker build -t ${IMAGE_NAME} .
fi

# 도커 컨테이너 실행
echo "도커 컨테이너 실행 중..."
docker run --rm \
    --device /dev/video0 \
    -e DISPLAY=${DISPLAY} \
    -v /tmp/.X11-unix:/tmp/.X11-unix \
    --env="QT_X11_NO_MITSHM=1" \
    --network host \
    -it ${IMAGE_NAME} bash
