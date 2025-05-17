@echo off
REM Script to run 3 Python scripts sequentially in a conda environment
REM Usage: run_python_scripts.bat.bat [conda_env_name]

setlocal enabledelayedexpansion

REM Check if environment name is provided
if "%1"=="" (
    echo No conda environment specified, using base environment.
    set ENV_NAME=base
) else (
    set ENV_NAME=%1
)

REM Activate the conda environment
call conda activate %ENV_NAME%
if %ERRORLEVEL% neq 0 (
    echo Failed to activate conda environment "%ENV_NAME%".
    echo Please make sure the environment exists.
    echo Available environments:
    conda env list
    exit /b 1
)

echo Using conda environment: %ENV_NAME%

REM Define the scripts to run
set SCRIPT1=src/lgb_tuning.py
set SCRIPT2=src/xgb_tuning.py
set SCRIPT3=src/catboost_implementation.py

REM Verify that all scripts exist
for %%s in (%SCRIPT1% %SCRIPT2% %SCRIPT3%) do (
    if not exist %%s (
        echo Error: Script %%s not found in the current directory.
        exit /b 1
    )
)

echo Starting scripts sequentially...

REM Run the scripts one by one
echo.
echo Running %SCRIPT1%...
python %SCRIPT1%
echo %SCRIPT1% completed.
echo.

echo Running %SCRIPT2%...
python %SCRIPT2%
echo %SCRIPT2% completed.
echo.

echo Running %SCRIPT3%...
python %SCRIPT3%
echo %SCRIPT3% completed.
echo.

echo All scripts have been executed successfully.

REM Deactivate conda environment when done
call conda deactivate

echo.
echo Press any key to exit...
pause > nul

endlocal
