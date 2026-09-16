@echo off
REM ============================================================
REM PGMT 本机 Windows 开发环境（纯逻辑单测，不含 Isaac Gym）
REM 用法: 在「已初始化 conda 的」cmd 中运行 setup\install_local.bat
REM
REM 版本对齐说明：本机 torch 固定 2.1.2（CPU），与服务器 PP4 环境的
REM torch 2.1.2+cu121 保持同版本，避免本机/服务器数值行为漂移。
REM ============================================================
setlocal
set ENV_NAME=pgmt-dev

where conda >nul 2>nul
if errorlevel 1 (
  echo [FAIL] 未找到 conda。请先安装 Miniconda/Anaconda 并确保 PATH 中有 conda。
  exit /b 1
)

call conda env list | findstr /C:"%ENV_NAME% " >nul
if errorlevel 1 (
  echo [i] 创建 conda 环境 %ENV_NAME% (python 3.10) ...
  call conda create -n %ENV_NAME% python=3.10 -y
  if errorlevel 1 (
    echo [FAIL] conda create 失败
    exit /b 1
  )
) else (
  echo [i] conda 环境 %ENV_NAME% 已存在，复用
)

call conda activate %ENV_NAME%
if errorlevel 1 (
  echo [FAIL] conda activate 失败。请先运行 "conda init cmd.exe" 并重开终端。
  exit /b 1
)

REM 确认激活生效：裸 pip 在激活失败时会写进 base 环境
python -c "import sys; print(sys.executable)" | findstr /I "%ENV_NAME%" >nul
if errorlevel 1 (
  echo [FAIL] 当前 python 不属于 %ENV_NAME% 环境，已中止（避免污染 base 环境）。
  echo        当前解释器:
  python -c "import sys; print(sys.executable)"
  exit /b 1
)

echo [i] 安装 torch 2.1.2 (CPU，与服务器同版本号) ...
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 exit /b 1

echo [i] 安装本机开发/测试依赖 ...
python -m pip install -r "%~dp0requirements-dev.txt"
if errorlevel 1 exit /b 1

echo.
echo [i] 冒烟：跑本机测试套件 ...
pushd "%~dp0.."
python -m pytest tests -q
set TEST_RC=%errorlevel%
popd
if not "%TEST_RC%"=="0" (
  echo [FAIL] 测试未全绿（exit %TEST_RC%）
  exit /b %TEST_RC%
)

echo.
echo [OK] 环境就绪且测试全绿。日常用法:
echo   conda activate %ENV_NAME% ^&^& cd /d D:\PGMT ^&^& python -m pytest tests -q
endlocal
