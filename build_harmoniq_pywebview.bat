@echo off
setlocal EnableExtensions EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem Configuration
rem ---------------------------------------------------------------------------
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%"
set "FRONTEND_DIR=%REPO_ROOT%harmoniq"
set "BUILD_ROOT=%REPO_ROOT%build\harmoniq_pywebview"
set "STAGING_DIR=%BUILD_ROOT%\staging"
set "ASSET_STAGING=%STAGING_DIR%\harmoniq_dist"
set "STAGED_DIST_NAME=harmoniq_dist"
set "VENV_DIR=%BUILD_ROOT%\venv"
set "REQUIREMENTS_FILE=%FRONTEND_DIR%\pywebview_requirements.txt"
set "ENTRY_SCRIPT=%REPO_ROOT%harmoniq_pywebview.py"
set "COMPILER_ROOT=%BUILD_ROOT%\compiler"
set "MUSIC_DIR=%REPO_ROOT%music"
set "MUSIC_STAGED_NAME=music"
set "LOG_PREFIX=[HARMONIQ EXE]"

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
call :log "Preparing Harmoniq PyWebview build"
call :require_command python || goto :error
call :require_command npm || goto :error

if not exist "%ENTRY_SCRIPT%" (
    echo %LOG_PREFIX% ERROR: Could not locate %ENTRY_SCRIPT%
    goto :error
)

if not exist "%REQUIREMENTS_FILE%" (
    echo %LOG_PREFIX% ERROR: Could not locate %REQUIREMENTS_FILE%
    goto :error
)

rem Ensure build directories exist
if not exist "%BUILD_ROOT%" mkdir "%BUILD_ROOT%"
if not exist "%STAGING_DIR%" mkdir "%STAGING_DIR%"
if not exist "%MUSIC_DIR%" mkdir "%MUSIC_DIR%"

rem ---------------------------------------------------------------------------
rem Install and build the Harmoniq frontend bundle
rem ---------------------------------------------------------------------------
call :log "Installing Harmoniq npm dependencies"
pushd "%FRONTEND_DIR%"
if not exist "node_modules" (
    npm install
    call :check_result "npm install failed"
) else (
    call :log "npm dependencies already installed"
)

call :log "Building Harmoniq production bundle"
npm run build
call :check_result "npm run build failed"
popd

if not exist "%FRONTEND_DIR%\dist" (
    echo %LOG_PREFIX% ERROR: Harmoniq build artefacts not found in %FRONTEND_DIR%\dist
    goto :error
)

rem ---------------------------------------------------------------------------
rem Stage static assets for PyInstaller
rem ---------------------------------------------------------------------------
call :log "Staging Harmoniq static assets"
if exist "%ASSET_STAGING%" (
    rmdir /s /q "%ASSET_STAGING%"
)
mkdir "%ASSET_STAGING%"
robocopy "%FRONTEND_DIR%\dist" "%ASSET_STAGING%" /mir >nul
set "ROBOCOPY_EXIT=%ERRORLEVEL%"
if %ROBOCOPY_EXIT% GEQ 8 (
    echo %LOG_PREFIX% ERROR: Failed to copy Harmoniq assets (robocopy code %ROBOCOPY_EXIT%)
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
    --name Harmoniq ^
    --pyinstaller-arg=--add-data ^
    --pyinstaller-arg="%ASSET_STAGING%;%STAGED_DIST_NAME%" ^
    --pyinstaller-arg=--add-data ^
    --pyinstaller-arg="%MUSIC_DIR%;%MUSIC_STAGED_NAME%"
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
