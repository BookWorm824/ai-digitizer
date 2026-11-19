@echo off
title AI Embroidery Digitizer
cd /d "%~dp0"

rem --- 1. Get the newest version from GitHub. Skipped quietly when offline. ---
if not exist ".git" goto :setup
where git >nul 2>nul
if errorlevel 1 goto :setup
echo Checking for updates...
git -c http.lowSpeedLimit=1000 -c http.lowSpeedTime=10 pull --ff-only --quiet >nul 2>nul

:setup
rem --- 2. First run only: make the app's own private Python. ---
if exist ".venv\Scripts\python.exe" goto :packages
echo Setting things up for the first time. This takes a few minutes...
py -3.12 -m venv .venv 2>nul || python -m venv .venv
if exist ".venv\Scripts\python.exe" goto :packages
echo.
echo Python 3.12 is needed but was not found. Install it from python.org, then run this again.
pause
exit /b 1

:packages
rem --- 3. Install what the app needs, the first time and whenever an update changes the list. ---
fc /b requirements.txt ".venv\requirements.installed" >nul 2>nul
if not errorlevel 1 goto :run
echo Installing what the app needs. This can take a few minutes...
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
if errorlevel 1 goto :failed
copy /y requirements.txt ".venv\requirements.installed" >nul
goto :run

:failed
echo.
echo Setup could not finish. Check your internet connection and run this again.
pause
exit /b 1

:run
".venv\Scripts\python.exe" -m app
if errorlevel 1 pause
