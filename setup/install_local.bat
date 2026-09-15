@echo off
REM ============================================================
REM PGMT 本机 Windows 开发环境（仅逻辑单测，不含 Isaac Gym）
REM 用法: 双击或在 cmd 中运行 install_local.bat
REM ============================================================
setlocal
set ENV_NAME=pgmt-dev

call conda env list | findstr /C:"%ENV_NAME% " >nul
if errorlevel 1 (
  echo [i] 创建 conda 环境 %ENV_NAME% (python 3.10) ...
  call conda create -n %ENV_NAME% python=3.10 -y
) else (
  echo [i] conda 环境 %ENV_NAME% 已存在，复用
)

call conda activate %ENV_NAME%
echo [i] 安装 torch CPU 版 ...
pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
echo [i] 安装测试依赖 ...
pip install numpy==1.26.4 pytest pyyaml rsl-rl-lib==2.1.2

echo.
echo [OK] 环境就绪。运行测试:
echo   conda activate %ENV_NAME% ^&^& cd /d D:\PGMT ^&^& pytest tests -v
endlocal
