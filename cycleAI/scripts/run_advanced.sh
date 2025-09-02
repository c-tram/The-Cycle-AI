#!/bin/bash

# Advanced Baseball AI Agent Launcher
# Run the LangChain-enhanced version of your baseball agent

echo "⚾ Advanced Baseball AI Agent"
echo "=============================="
echo ""

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Check if advanced dependencies are installed
echo "📦 Checking advanced dependencies..."
python -c "import langchain" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  LangChain not found! Installing advanced dependencies..."
    pip install -r requirements_advanced.txt
fi

echo ""
echo "🚀 Starting Advanced Baseball Agent..."
echo "======================================"
echo ""
echo "Available modes:"
echo "• Simple (Fast) - Your original system"
echo "• LangChain (Advanced) - Smart tool selection"
echo "• Auto (Smart) - Chooses best approach"
echo ""
echo "Use the radio buttons in the sidebar to switch modes!"
echo ""

# Run the advanced agent
streamlit run langchain_baseball_agent.py --server.headless true --server.port 8501

# Deactivate virtual environment when done
deactivate
