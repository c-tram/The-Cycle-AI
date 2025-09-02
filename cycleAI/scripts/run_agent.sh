#!/bin/bash

# Baseball AI Agent Launcher
# Choose between Simple and Advanced versions

echo "⚾ Baseball AI Agent Launcher"
echo "=============================="
echo ""
echo "Choose your version:"
echo "1) Simple (Fast) - Original API-only system"
echo "2) Advanced (Smart) - LangChain-enhanced with tools"
echo "3) Auto (Hybrid) - Smart switching between modes"
echo ""
read -p "Enter choice (1-3): " choice

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "Run: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

case $choice in
    1)
        echo ""
        echo "🚀 Starting SIMPLE Baseball Agent..."
        echo "====================================="
        streamlit run app.py --server.headless true --server.port 8501
        ;;
    2)
        echo ""
        echo "🚀 Starting ADVANCED Baseball Agent..."
        echo "======================================"
        # Check if advanced dependencies are installed
        python -c "import langchain" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "⚠️  Installing advanced dependencies..."
            pip install -r requirements_advanced.txt
        fi
        streamlit run langchain_baseball_agent.py --server.headless true --server.port 8501
        ;;
    3)
        echo ""
        echo "🚀 Starting HYBRID Baseball Agent..."
        echo "===================================="
        # Check if advanced dependencies are installed
        python -c "import langchain" 2>/dev/null
        if [ $? -ne 0 ]; then
            echo "⚠️  Installing advanced dependencies..."
            pip install -r requirements_advanced.txt
        fi
        streamlit run langchain_baseball_agent.py --server.headless true --server.port 8501
        ;;
    *)
        echo "❌ Invalid choice. Please run again and choose 1, 2, or 3."
        exit 1
        ;;
esac

# Deactivate virtual environment when done
deactivate
