@echo off
setlocal
title Scholarship Management - Kolehiyo ng Subic
cd /d "%~dp0"

echo ==============================================
echo  Scholarship Management - Scholarship Application and
echo  Performance Monitoring Platform
echo  Kolehiyo ng Subic
echo ==============================================
echo.

where python >nul 2>nul
if errorlevel 1 goto no_python

python -c "import flask, sklearn, pandas" >nul 2>nul
if errorlevel 1 goto install_deps
goto check_model

:no_python
echo [ERROR] Python was not found. Please install Python 3.10 or later.
pause
exit /b 1

:install_deps
echo Installing dependencies - this is needed only on first run.
echo.
python -m pip install -r requirements.txt
if errorlevel 1 goto pip_fail
goto check_model

:pip_fail
echo [ERROR] Failed to install dependencies.
pause
exit /b 1

:check_model
if exist "ml\artifacts\retention_model.joblib" goto start_server

:train_model
echo Training retention classification model. Please wait.
echo.
python -m ml.train_model
if errorlevel 1 goto train_fail
goto start_server

:train_fail
echo [ERROR] Model training failed.
pause
exit /b 1

:start_server
set FLASK_USE_RELOADER=0
echo.
echo Starting server at http://127.0.0.1:5000
echo A browser tab will open automatically once the server is ready.
echo Press Ctrl+C in this window to stop the server.
echo.
python app.py

echo.
echo Server has stopped.
pause
