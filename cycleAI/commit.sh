#!/bin/bash

# Intelligent Baseball AI - Git Commit Script
# This script automates the process of committing and pushing changes to GitHub

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in a git repository
if ! git rev-parse --git-dir > /dev/null 2>&1; then
    print_error "Not a git repository. Please run this script from the project root."
    exit 1
fi

# Check if we're in the cycleAI directory and adjust path if needed
if [[ "$(basename $(pwd))" == "cycleAI" ]]; then
    print_status "Detected cycleAI directory. Adding only cycleAI files..."
    # Only add files in the current directory (cycleAI)
    git add .
else
    print_warning "Not in cycleAI directory. Adding all changes..."
    git add .
fi

# Get commit message
if [ $# -eq 0 ]; then
    # No arguments provided, use comprehensive update message
    COMMIT_MSG="🚀 BETA: Major System Refactor: Multi-Modal Baseball AI with Autonomous Capabilities

✨ New Features:
- 🤖 Autonomous Prompt Generation System
- 📊 Interactive Analytics Dashboard  
- 🧠 Advanced LangChain Integration
- 🏆 Best Team Builder with Explanations

🏗️ Architecture Changes:
- Organized into Simple/Advanced/Autonomous modes
- Clean directory structure with shared utilities
- Proper import paths and module organization
- Launcher scripts for each mode

🔧 System Improvements:
- Enhanced error handling and position filtering
- Quality scoring for automated prompt testing
- Performance monitoring and insights
- Comprehensive documentation updates

Built $(date '+%Y-%m-%d %H:%M:%S')"
    print_warning "Using comprehensive update message for all recent work"
else
    # Use provided arguments as commit message
    COMMIT_MSG="$*"
fi

# Check git status
if [[ -z $(git status --porcelain) ]]; then
    print_warning "No changes to commit."
    exit 0
fi

print_status "Checking git status..."
git status --short

# Commit with message
print_status "Committing changes..."
if git commit -m "$COMMIT_MSG"; then
    print_success "Changes committed successfully!"
else
    print_error "Failed to commit changes."
    exit 1
fi

# Check if remote exists and push
if git remote get-url origin > /dev/null 2>&1; then
    print_status "Pushing to remote repository..."
    if git push origin $(git branch --show-current); then
        print_success "Changes pushed to GitHub successfully!"
        print_status "Commit URL: $(git remote get-url origin | sed 's/\.git$//')/commit/$(git rev-parse HEAD)"
    else
        print_error "Failed to push changes. You may need to push manually."
        print_status "Current branch: $(git branch --show-current)"
        print_status "Run: git push origin $(git branch --show-current)"
    fi
else
    print_warning "No remote repository configured. Changes committed locally only."
fi

print_success "Git operations completed!"
echo ""
print_status "Summary:"
echo "  - Commit: $(git rev-parse --short HEAD)"
echo "  - Branch: $(git branch --show-current)"
echo "  - Message: $COMMIT_MSG"
