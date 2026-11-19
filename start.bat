@echo off
title AI Embroidery Digitizer
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
  echo Setting things up for the first time. This takes a few minutes...
  py -3.12 -m venv .venv 2>nul || python -m venv .venv
  if not exist ".venv\Scripts\python.exe" (
    echo.
    echo Python 3.12 is needed but was not found. Install it from python.org, then run this again.
    pause
    exit /b 1
  )
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -q -r requirements.txt
  if errorlevel 1 (
    echo.
    echo Setup could not finish. Check your internet connection and run this again.
    rmdir /s /q .venv
    pause
    exit /b 1
  )
)

".venv\Scripts\python.exe" -m app
if errorlevel 1 pause
