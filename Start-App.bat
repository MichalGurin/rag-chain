@echo off
powershell.exe -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%~dp0start-app.ps1"
if errorlevel 1 (
    echo.
    echo Application startup did not complete. Review the error above.
    pause
)
