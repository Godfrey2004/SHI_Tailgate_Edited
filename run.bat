@echo off
set "PATH=C:\Users\HAI\AppData\Local\Programs\Python\Python310;C:\Users\HAI\AppData\Local\Programs\Python\Python310\Scripts;%PATH%"
cd /d "%~dp0"
echo Starting SHI Tail Gate application...
start /b python -c "import time, webbrowser; time.sleep(2); webbrowser.open('http://localhost:5000')"
python app.py
pause
