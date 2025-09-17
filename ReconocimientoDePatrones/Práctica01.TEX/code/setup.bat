@echo off␍
echo ==========================================␍
echo    LOVDOG TOOLKIT SETUP SCRIPT␍
echo ==========================================␍
␍
:: Check if Python is installed␍
python --version >nul 2>&1␍
if errorlevel 1 (␍
    echo ❌ Python is not installed. Please install Python 3.8 or higher.␍
    pause␍
    exit /b 1␍
)␍
␍
:: Create virtual environment␍
echo Creating virtual environment...␍
python -m venv venv␍
␍
if errorlevel 1 (␍
    echo ❌ Failed to create virtual environment.␍
    pause␍
    exit /b 1␍
)␍
␍
:: Activate virtual environment␍
echo Activating virtual environment...␍
call venv\Scripts\activate.bat␍
␍
:: Upgrade pip␍
echo Upgrading pip...␍
python -m pip install --upgrade pip␍
␍
:: Install requirements␍
echo Installing requirements...␍
if exist "res\requirements.txt" (␍
    pip install -r res\requirements.txt␍
) else (␍
    echo ❌ requirements.txt not found in res directory.␍
    echo Installing default packages...␍
    pip install opencv-python scikit-image scikit-learn pandas numpy matplotlib␍
)␍
␍
:: Create directories␍
echo Creating output directories...␍
mkdir results 2>nul␍
mkdir features 2>nul␍
mkdir models 2>nul␍
mkdir plots 2>nul␍
␍
echo ==========================================␍
echo    SETUP COMPLETED SUCCESSFULLY! 🎉␍
echo ==========================================␍
echo.␍
echo Next steps:␍
echo 1. Activate the virtual environment:␍
echo    venv\Scripts\activate.bat␍
echo.␍
echo 2. Test the setup:␍
echo    python res\quick_test.py␍
echo.␍
echo 3. Run the full workflow:␍
echo    python res\main.py␍
echo.␍
echo 4. Deactivate when done:␍
echo    deactivate␍
echo ==========================================␍
pause@echo off
echo ==========================================
echo    LOVDOG TOOLKIT SETUP SCRIPT
echo ==========================================

:: Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

:: Create virtual environment
echo Creating virtual environment...
python -m venv venv

if errorlevel 1 (
    echo ❌ Failed to create virtual environment.
    pause
    exit /b 1
)

:: Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

:: Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip

:: Install requirements
echo Installing requirements...
if exist "res\requirements.txt" (
    pip install -r res\requirements.txt
) else (
    echo ❌ requirements.txt not found in res directory.
    echo Installing default packages...
    pip install opencv-python scikit-image scikit-learn pandas numpy matplotlib
)

:: Create directories
echo Creating output directories...
mkdir results 2>nul
mkdir features 2>nul
mkdir models 2>nul
mkdir plots 2>nul

echo ==========================================
echo    SETUP COMPLETED SUCCESSFULLY! 🎉
echo ==========================================
echo.
echo Next steps:
echo 1. Activate the virtual environment:
echo    venv\Scripts\activate.bat
echo.
echo 2. Test the setup:
echo    python res\quick_test.py
echo.
echo 3. Run the full workflow:
echo    python res\main.py
echo.
echo 4. Deactivate when done:
echo    deactivate
echo ==========================================
pause
