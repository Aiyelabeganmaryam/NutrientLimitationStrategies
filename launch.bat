@echo off
REM Launch script for Nutrient Limitation Strategies & Spatial Analysis (Windows)

echo 🌱 Nutrient Limitation Strategies ^& Spatial Analysis
echo ==================================================
echo Advanced Spatial Techniques for Agricultural Nutrient Management
echo.

REM Check if we're in the right directory
if not exist "streamlit_app\main.py" (
    echo ❌ Error: Please run this script from the NutrientLimitationStrategies directory
    echo Expected file structure:
    echo   NutrientLimitationStrategies\
    echo   ├── streamlit_app\
    echo   │   ├── main.py
    echo   │   └── requirements.txt
    echo   └── launch.bat
    pause
    exit /b 1
)

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ Python is not installed. Please install Python 3.8 or higher.
    pause
    exit /b 1
)

echo ✅ Python found:
python --version

REM Check if virtual environment exists
if not exist "nutrient_env" (
    echo 📦 Creating virtual environment...
    python -m venv nutrient_env
) else (
    echo ✅ Virtual environment found
)

REM Activate virtual environment
echo 🔧 Activating virtual environment...
call nutrient_env\Scripts\activate

REM Check if Streamlit is installed
pip show streamlit >nul 2>&1
if errorlevel 1 (
    echo 📥 Installing required packages...
    pip install -r streamlit_app\requirements.txt
) else (
    echo ✅ Dependencies are installed
)

echo.
echo 🚀 Starting Nutrient Limitation Strategies ^& Spatial Analysis...
echo 🌐 The application will open in your default browser
echo 📊 Features: Moran's I analysis, LISA clustering, spatial mapping
echo.

REM Start the Streamlit application
cd streamlit_app
streamlit run main.py

echo.
echo 👋 Application stopped. Thank you for using Nutrient Limitation Strategies!
pause