#!/bin/bash

# Autonomous Prompt Generator Streamlit Interface
echo "🤖 Autonomous Baseball Prompt Generator (Streamlit Interface)"
echo "============================================================="

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source .venv/bin/activate

# Install additional dependencies for visualization
echo "📦 Installing visualization dependencies..."
pip install plotly pandas > /dev/null 2>&1

echo ""
echo "🚀 Starting Autonomous Prompt Generator Interface..."
echo ""
echo "This interface provides:"
echo "• 🎲 Batch prompt generation with live progress"
echo "• 📊 Real-time analytics and performance charts" 
echo "• 🎯 Best prompt discovery and ranking"
echo "• 📚 Comprehensive prompt library management"
echo "• 🔬 Live prompt testing and evaluation"
echo ""

# Start Streamlit app
streamlit run autonomous_prompt_app.py --server.port 8502

echo ""
echo "🎉 Interface closed!"
