@echo off
set PYTHON="C:\Users\user\Downloads\yolov8_trashdetectionproject-main\yolov8_trashdetectionproject-main\.venv\Scripts\python.exe"

echo ===================================================
echo     Floating Trash Detection - YOLOv8
echo ===================================================
echo.
echo Choose an option:
echo   [1] Launch Interactive Web UI (Gradio)
echo   [2] Run Detection on Test Images (Save to predictions/)
echo   [3] Exit
echo.
set /p opt="Enter choice (1/2/3): "

if "%opt%"=="1" (
    echo Starting Gradio Web App...
    %PYTHON% app.py
) else if "%opt%"=="2" (
    echo Running Batch Detection...
    %PYTHON% predict.py
    pause
) else (
    exit /b
)
