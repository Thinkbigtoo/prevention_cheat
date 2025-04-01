@echo off
set IMAGE_NAME=gaze-tracking

echo 도커 이미지 확인 중...
docker images | findstr "%IMAGE_NAME%" >nul
if errorlevel 1 (
    echo 도커 이미지 빌드 중...
    docker build -t %IMAGE_NAME% .
) else (
    echo 도커 이미지가 이미 존재합니다.
)

echo 도커 컨테이너 실행 중...
docker run --rm ^
    --device /dev/video0 ^
    -e DISPLAY=%DISPLAY% ^
    -v /tmp/.X11-unix:/tmp/.X11-unix ^
    --env="QT_X11_NO_MITSHM=1" ^
    --network host ^
    -it %IMAGE_NAME% bash

pause 