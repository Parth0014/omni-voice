@echo off
title OmniVoice - Gratitude Story Narration
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0scripts\run-local-test.ps1"
set "RUN_EXIT=%errorlevel%"
echo.
pause
exit /b %RUN_EXIT%
