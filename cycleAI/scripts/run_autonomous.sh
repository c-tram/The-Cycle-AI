#!/bin/bash

# Autonomous Prompt Generator Launcher
echo "🤖 Autonomous Baseball Prompt Generator"
echo "========================================"

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if required dependencies are installed
echo "📦 Checking dependencies..."
python -c "import langchain, openai, streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements_advanced.txt
fi

echo ""
echo "🚀 Starting Autonomous Prompt Generation..."
echo ""
echo "This will:"
echo "• Generate diverse baseball prompts automatically"
echo "• Test them against your baseball agent"
echo "• Evaluate response quality"
echo "• Build a comprehensive prompt library"
echo "• Provide performance insights"
echo ""

# Run the autonomous generator
python autonomous_prompt_generator.py

echo ""
echo "🎉 Autonomous generation complete!"
echo "📋 Check autonomous_prompt_library.json for results"
