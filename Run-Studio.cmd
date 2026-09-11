@echo off
title OmniVoice Studio
cd /d "%~dp0"
"%~dp0.venv\Scripts\python.exe" -u "%~dp0local_ui.py"
if errorlevel 1 pause
