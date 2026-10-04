@echo off
echo ========================================================
echo Starting DIF-SEG Brain Tumor Analysis
echo ========================================================
echo.

if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found. 
    echo Please run setup.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate

echo.
echo Starting Streamlit Application...
python -m streamlit run app.py

pause
