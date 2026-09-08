@echo off
setlocal EnableDelayedExpansion
REM ============================================================
REM operator-100 Windows dev helper.
REM Loads the MSVC environment (vcvars64) and puts the Python
REM environment dirs on PATH, so torch cpp_extension can find
REM cl.exe and ninja.
REM
REM Usage:   tools\run_with_msvc.bat <command...>
REM Example: tools\run_with_msvc.bat D:\conda_envs\dl\python.exe -m pytest -q
REM Note:    if nvcc complains about "unsupported Microsoft Visual
REM          Studio version", run first:
REM          set TORCH_NVCC_FLAGS=-allow-unsupported-compiler
REM ============================================================
if "%~1"=="" (
  echo Usage: %~nx0 ^<command...^>
  echo Example: %~nx0 D:\conda_envs\dl\python.exe -m pytest -q
  exit /b 1
)

REM 1) If cl.exe is not on PATH, locate Visual Studio via vswhere and load vcvars64
where cl >nul 2>&1
if errorlevel 1 (
  for /f "usebackq tokens=*" %%i in (`"%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe" -latest -products * -property installationPath 2^>nul`) do (
    call "%%i\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1
  )
)

REM 2) If the first arg is an exe under a python env, add Scripts / Library\bin to PATH (for ninja)
if /i "%~x1"==".exe" if exist "%~dp1Scripts" (
  set "PATH=%~dp1;%~dp1Scripts;%~dp1Library\bin;%PATH%"
)

%*
