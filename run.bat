@echo off
title SHI Tailgate Inspection System
cd /d "%~dp0"
echo ============================================================
echo   SHI Tailgate Inspection System
echo   Starting server... Please wait.
echo ============================================================
start /b python -c "import time, webbrowser; time.sleep(3); webbrowser.open('http://localhost:5000')"
python app.py
pause
