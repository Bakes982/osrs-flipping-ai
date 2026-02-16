@echo off
echo ============================================================
echo DINK Webhook Receiver - Starting...
echo ============================================================
echo.
echo Make sure you have:
echo 1. DINK plugin installed in RuneLite
echo 2. DINK webhook URL set to: http://localhost:5000/dink
echo 3. Grand Exchange notifications enabled in DINK settings
echo.
echo Press Ctrl+C to stop the server
echo ============================================================
echo.
cd /d "%~dp0"
python dink_receiver.py
pause
