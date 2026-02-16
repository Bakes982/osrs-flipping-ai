@echo off
echo ========================================
echo OSRS Flip Bot Starting...
echo ========================================
echo.

cd "C:\Users\Mikeb\OneDrive\Desktop\Flipping AI"

echo Installing dependencies...
pip install pandas scikit-learn requests numpy joblib --quiet

echo.
echo Starting live price monitor...
echo Press Ctrl+C to stop
echo.

python live_price_monitor.py --continuous

pause