@echo off
title NETRADRISHTI - Tele-Triage System (SIH 2026)
cd /d "%~dp0"
echo ===================================================================
echo   NETRADRISHTI: Explainable AI for Diabetic Retinopathy (SIH 2026)
echo   Starting FastAPI Application Server...
echo ===================================================================
echo.
echo Opening web browser at http://localhost:8000 ...
start "" http://localhost:8000
echo.
echo Launching Uvicorn server (Press Ctrl+C to stop)...
python -m uvicorn server:app --host 127.0.0.1 --port 8000 --reload
pause
