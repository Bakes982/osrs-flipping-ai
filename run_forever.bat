@echo off
:: ========================================
::  OSRS Flipping AI - 24/7 Watchdog
::  Auto-restarts backend if it crashes.
::  Logs output to logs\backend.log
:: ========================================

C:
cd "C:\Users\Mikeb\OneDrive\Desktop\Flipping AI"

:: Create logs directory
if not exist "logs" mkdir "logs"

:: Install deps and init DB on first run
echo [%date% %time%] Installing dependencies...
pip install -r requirements.txt --quiet 2>nul

echo [%date% %time%] Initializing database...
python -m backend.migrate 2>>"logs\migrate.log"

:: Build frontend if not already built
if not exist "frontend\dist\index.html" (
    echo [%date% %time%] Building frontend...
    cd "C:\Users\Mikeb\OneDrive\Desktop\Flipping AI\frontend"
    call npm install --silent 2>nul
    call npm run build 2>>"C:\Users\Mikeb\OneDrive\Desktop\Flipping AI\logs\frontend_build.log"
    cd "C:\Users\Mikeb\OneDrive\Desktop\Flipping AI"
)

echo.
echo ========================================
echo  Flipping AI - 24/7 Mode
echo  Dashboard: http://localhost:8001
echo  Logs:      logs\backend.log
echo  Press Ctrl+C to stop
echo ========================================
echo.

:: Watchdog loop - restart on crash
:loop
echo [%date% %time%] Starting backend... >> "logs\backend.log"
echo [%date% %time%] Starting backend...

:: Run uvicorn in foreground (no --reload for stability in production)
python -m uvicorn backend.app:app --host 0.0.0.0 --port 8001 --log-level info

echo.
echo [%date% %time%] Backend stopped! Restarting in 5 seconds... >> "logs\backend.log"
echo [%date% %time%] Backend stopped! Restarting in 5 seconds...
ping 127.0.0.1 -n 6 >nul
goto loop
