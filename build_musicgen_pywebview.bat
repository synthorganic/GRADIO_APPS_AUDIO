@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem Configuration
rem ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%"
set "BUILD_ROOT=%REPO_ROOT%build\musicgen_pywebview"
set "STAGING_DIR=%BUILD_ROOT%\staging"
set "ASSET_SOURCE=%REPO_ROOT%assets"
set "ASSET_STAGING=%STAGING_DIR%\assets"
set "NOTEBOOK_SOURCE=%REPO_ROOT%prompt_notebook.csv"
set "NOTEBOOK_STAGING=%STAGING_DIR%\prompt_notebook.csv"
set "VENV_DIR=%BUILD_ROOT%\venv"
set "ENTRY_SCRIPT=%REPO_ROOT%musicgen_pywebview.py"
set "REQUIREMENTS_FILE=%REPO_ROOT%musicgen_pywebview_requirements.txt"
set "COMPILER_ROOT=%BUILD_ROOT%\compiler"
set "LOG_PREFIX=[MUSICGEN EXE]"

rem ---------------------------------------------------------------------------
rem Helper routines
rem ---------------------------------------------------------------------------
:log
    echo %LOG_PREFIX% %~1
    exit /b 0

:require_command
    where %~1 >nul 2>&1
    if errorlevel 1 (
        echo %LOG_PREFIX% ERROR: Required command "%~1" not found in PATH.
        exit /b 1
    )
    exit /b 0

:check_result
    if errorlevel 1 (
        echo %LOG_PREFIX% ERROR: %~1
        goto :error
    )
    exit /b 0

rem ---------------------------------------------------------------------------
rem Pre-flight validation
rem ---------------------------------------------------------------------------
call :log "Preparing MusicGen PyWebview build"
call :require_command python || goto :error

if not exist "%ENTRY_SCRIPT%" (
    echo %LOG_PREFIX% ERROR: Could not locate %ENTRY_SCRIPT%
    goto :error
)

if not exist "%REQUIREMENTS_FILE%" (
    echo %LOG_PREFIX% ERROR: Could not locate %REQUIREMENTS_FILE%
    goto :error
)

if not exist "%ASSET_SOURCE%" (
    echo %LOG_PREFIX% ERROR: Could not locate assets directory at %ASSET_SOURCE%
    goto :error
)

if not exist "%NOTEBOOK_SOURCE%" (
    echo %LOG_PREFIX% ERROR: Could not locate prompt notebook at %NOTEBOOK_SOURCE%
    goto :error
)

if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%"
if not exist "%STAGING_DIR%" mkdir "%STAGING_DIR%"

rem ---------------------------------------------------------------------------
rem Stage static assets for PyInstaller
rem ---------------------------------------------------------------------------
call :log "Staging MusicGen assets"
if exist "%ASSET_STAGING%" (
    rmdir /s /q "%ASSET_STAGING%"
)
mkdir "%ASSET_STAGING%"
robocopy "%ASSET_SOURCE%" "%ASSET_STAGING%" /mir >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if %ROBOCOPY_EXIT% GEQ 8 (
    echo %LOG_PREFIX% ERROR: Failed to copy MusicGen assets (robocopy code %ROBOCOPY_EXIT%)
    goto :error
)

copy /y "%NOTEBOOK_SOURCE%" "%NOTEBOOK_STAGING%" >nul
if errorlevel 1 (
    echo %LOG_PREFIX% ERROR: Failed to copy prompt notebook
    goto :error
)

rem ---------------------------------------------------------------------------
rem Prepare Python environment
rem ---------------------------------------------------------------------------
call :log "Ensuring Python virtual environment"
if not exist "%VENV_DIR%\Scripts\python.exe" (
    python -m venv "%VENV_DIR%"
    call :check_result "Failed to create virtual environment"
)
call "%VENV_DIR%\Scripts\activate.bat"
if errorlevel 1 (
    echo %LOG_PREFIX% ERROR: Failed to activate virtual environment
    goto :error
)

call :log "Upgrading pip"
python -m pip install --upgrade pip >nul
call :check_result "pip upgrade failed"

call :log "Installing PyWebview dependencies"
pip install -r "%REQUIREMENTS_FILE%"
call :check_result "pip install failed"

rem ---------------------------------------------------------------------------
rem Invoke the exe compiler helper
rem ---------------------------------------------------------------------------
call :log "Packaging PyInstaller executable"
python -m exe_compiler "%ENTRY_SCRIPT%" ^
    --requirements "%REQUIREMENTS_FILE%" ^
    --build-root "%COMPILER_ROOT%" ^
    --name "MusicGenStems" ^
    --pyinstaller-arg=--add-data ^
    --pyinstaller-arg="%ASSET_STAGING%;assets" ^
    --pyinstaller-arg=--add-data ^
    --pyinstaller-arg="%NOTEBOOK_STAGING%;."
call :check_result "PyInstaller build failed"

call :log "Executable available under %COMPILER_ROOT%\dist"
call :log "Build complete"

goto :success

:error
endlocal
exit /b 1

:success
endlocal
exit /b 0
