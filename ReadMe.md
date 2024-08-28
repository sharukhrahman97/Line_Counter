> Works only on python v3.10.1

# Mimick RTSP

## Windows
```
.\test\mediamtx.exe .\test\config1.yml

.\test\mediamtx.exe .\test\config2.yml
```
```
.\test\ffmpeg.exe -re -stream_loop -1 -i .\test\cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1

.\test\ffmpeg.exe -re -stream_loop -1 -i .\test\cam2.mp4 -c copy -f rtsp rtsp://localhost:8555/cam2
```

## Mac
```
.\test\mediamtx .\test\config1.yml

.\test\mediamtx .\test\config2.yml
```
```
.\test\ffmpeg -re -stream_loop -1 -i .\test\cam1.mp4 -c copy -f rtsp rtsp://localhost:8554/cam1

.\test\ffmpeg -re -stream_loop -1 -i .\test\cam2.mp4 -c copy -f rtsp rtsp://localhost:8555/cam2
```

# To install

> pip install --upgrade pip wheel setuptools
> pip install -r .\requirements.txt

# To run

> python main.py

`if you get this error message "OSError: [WinError 126] The specified module could not be found. Error loading "F:\queue\.venv\lib\site-packages\torch\lib\fbgemm.dll" or one of its dependencies." place the .\misc\libomp140.x86_64.dll to System32 folder`