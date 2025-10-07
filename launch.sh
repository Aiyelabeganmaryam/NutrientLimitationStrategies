#!/bin/bash
# Launch script for Nutrient Limitation Strategies & Spatial Analysis

echo "🌱 Nutrient Limitation Strategies & Spatial Analysis"
echo "=================================================="
echo "Advanced Spatial Techniques for Agricultural Nutrient Management"
echo ""

# Check if we're in the right directory
if [ ! -f "streamlit_app/main.py" ]; then
    echo "❌ Error: Please run this script from the NutrientLimitationStrategies directory"
    echo "Expected file structure:"
    echo "  NutrientLimitationStrategies/"
    echo "  ├── streamlit_app/"
    echo "  │   ├── main.py"
    echo "  │   └── requirements.txt"
    echo "  └── launch.sh"
    exit 1
fi

# Check if Python is available
if ! command -v python &> /dev/null; then
    echo "❌ Python is not installed. Please install Python 3.8 or higher."
    exit 1
fi

echo "✅ Python found: $(python --version)"

# Check if virtual environment exists
if [ ! -d "nutrient_env" ]; then
    echo "📦 Creating virtual environment..."
    python -m venv nutrient_env
else
    echo "✅ Virtual environment found"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source nutrient_env/bin/activate

# Check if Streamlit is installed
if ! pip show streamlit &> /dev/null; then
    echo "📥 Installing required packages..."
    pip install -r streamlit_app/requirements.txt
else
    echo "✅ Dependencies are installed"
fi

echo ""
echo "🚀 Starting Nutrient Limitation Strategies & Spatial Analysis..."
echo "🌐 The application will open in your default browser"
echo "📊 Features: Moran's I analysis, LISA clustering, spatial mapping"
echo ""

# Start the Streamlit application
cd streamlit_app
streamlit run main.py

echo ""
echo "👋 Application stopped. Thank you for using Nutrient Limitation Strategies!"