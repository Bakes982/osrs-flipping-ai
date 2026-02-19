@echo off
echo ============================================
echo   FlippingAI - RuneLite Plugin Launcher
echo ============================================
echo.
echo This launches the OFFICIAL RuneLite client
echo with the FlippingAI plugin pre-loaded.
echo.
echo Same RuneLite, same settings, same login.
echo Your existing plugins will all still work.
echo.
echo Close your current RuneLite first!
echo.
pause

cd /d "%~dp0runelite-plugin"
set JAVA_HOME=C:\Program Files\Eclipse Adoptium\jdk-21.0.7.6-hotspot
"C:\Users\Mikeb\.gradle\wrapper\dists\gradle-9.2.0-bin\11i5gvueggl8a5cioxuftxrik\gradle-9.2.0\bin\gradle.bat" runClient
