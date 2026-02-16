@echo off
echo ========================================
echo  OSRS Flipping AI - Starting Up
echo ========================================
echo.

:: Install Python deps
echo Installing Python dependencies...
pip install -r requirements.txt --quiet 2>nul

:: Initialize database
echo Initializing database...
python -m backend.migrate
echo.

:: Start FastAPI backend (port 8001)
echo Starting backend on http://localhost:8001 ...
start "FlippingAI-Backend" cmd /c "cd /d %~dp0 && uvicorn backend.app:app --host 0.0.0.0 --port 8001 --reload"

:: Wait for backend to start
timeout /t 3 /nobreak >nul

:: Start React frontend (port 5173)
echo Starting frontend on http://localhost:5173 ...
start "FlippingAI-Frontend" cmd /c "cd /d %~dp0\frontend && npm run dev"

:: Wait and open browser
timeout /t 3 /nobreak >nul
start http://localhost:5173

echo.
echo ========================================
echo  Flipping AI is running!
echo  Backend:  http://localhost:8001
echo  Frontend: http://localhost:5173
echo  API Docs: http://localhost:8001/docs
echo ========================================
echo.
echo Press any key to stop all services...
pause >nul

:: Kill processes
taskkill /FI "WINDOWTITLE eq FlippingAI-Backend" /F 2>nul
taskkill /FI "WINDOWTITLE eq FlippingAI-Frontend" /F 2>nul
