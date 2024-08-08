# Windows
```
.\test\mediamtx.exe .\test\config1.yml

.\test\mediamtx.exe .\test\config2.yml
```
```
.\test\ffmpeg.exe -re -stream_loop -1 -i .\test\cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1

.\test\ffmpeg.exe -re -stream_loop -1 -i .\test\cam2.mp4 -c copy -f rtsp rtsp://localhost:8555/cam2
```

# Mac
```
.\test\mediamtx .\test\config1.yml

.\test\mediamtx .\test\config2.yml
```
```
.\test\ffmpeg -re -stream_loop -1 -i .\test\cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1

.\test\ffmpeg -re -stream_loop -1 -i .\test\cam2.mp4 -c copy -f rtsp rtsp://localhost:8555/cam2
```

