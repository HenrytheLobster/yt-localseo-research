@echo off
cd /d C:\Users\naylo\automation\yt-kindle-research

echo Running YouTube Research Engine...
C:\Users\naylo\AppData\Local\Programs\Python\Python313\python.exe run_agent.py --mode full --max 200

echo.
echo Done.
pause
