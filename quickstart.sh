#!/usr/bin/env bash
# ---------------------------------------------------------------
# Quickstart script for WhatsApp Database Insights
# Works on macOS, Linux, and Windows (Git Bash/WSL)
# ---------------------------------------------------------------
set -e

REPO_URL="https://github.com/victor-gurbani/whatsapp-database-insights.git"
REPO_NAME="whatsapp-database-insights"

echo "🚀 WhatsApp Database Insights - Quickstart"
echo "==========================================="

# Check arguments
USE_VENV=true
for arg in "$@"; do
    if [ "$arg" == "--no-venv" ]; then
        USE_VENV=false
    fi
done

# Termux detection and setup
if [ -n "$TERMUX_VERSION" ]; then
    echo "🤖 Termux environment detected!"
    USE_VENV=false
    echo "📦 Installing required packages via pkg..."
    pkg install -y tur-repo
    pkg install -y python-numpy matplotlib python-pyarrow python-pandas
fi

# --- 1. Clone or navigate into the repo ---
if [ -d ".git" ] && git remote get-url origin 2>/dev/null | grep -q "$REPO_NAME"; then
    echo "✅ Already inside the repository"
elif [ -d "$REPO_NAME" ]; then
    echo "📂 Found existing $REPO_NAME folder, entering..."
    cd "$REPO_NAME"
elif [ -d "GitHub/$REPO_NAME" ]; then
    echo "📂 Found existing GitHub/$REPO_NAME folder, entering..."
    cd "GitHub/$REPO_NAME"
else
    echo "📥 Cloning repository..."
    git clone "$REPO_URL"
    cd "$REPO_NAME"
fi

# --- 2. Pull latest changes (fast-forward only) ---
echo "🔄 Pulling latest changes..."
git pull --ff-only || echo "⚠️  Pull failed or nothing to update (you may have local changes)"

# --- 3. Ensure pip works and is upgraded ---
echo "🐍 Checking Python & pip..."
PYTHON_CMD=""
if command -v python3 &>/dev/null; then
    PYTHON_CMD="python3"
elif command -v python &>/dev/null; then
    PYTHON_CMD="python"
else
    echo "❌ Python not found! Please install Python 3.8+ and try again."
    exit 1
fi

echo "   Using: $($PYTHON_CMD --version)"

# Upgrade pip
echo "📦 Upgrading pip..."
$PYTHON_CMD -m pip install --upgrade pip --quiet 2>/dev/null || \
$PYTHON_CMD -m ensurepip --upgrade --quiet 2>/dev/null && \
$PYTHON_CMD -m pip install --upgrade pip --quiet

# --- 4. Create or activate virtual environment ---
if [ "$USE_VENV" = true ]; then
    VENV_DIR=".venv"

    if [ ! -d "$VENV_DIR" ]; then
        echo "🔧 Creating virtual environment..."
        $PYTHON_CMD -m venv "$VENV_DIR"
    fi

    echo "🔌 Activating virtual environment..."
    # Cross-platform activation
    if [ -f "$VENV_DIR/bin/activate" ]; then
        source "$VENV_DIR/bin/activate"
    elif [ -f "$VENV_DIR/Scripts/activate" ]; then
        source "$VENV_DIR/Scripts/activate"
    else
        echo "❌ Could not find venv activation script"
        exit 1
    fi

    # Use venv python for subsequent commands
    PYTHON_CMD="python"

    # Upgrade pip inside venv
    $PYTHON_CMD -m pip install --upgrade pip --quiet
else
    echo "🚫 Skipping virtual environment setup (--no-venv flag detected)."
fi

# --- 5. Install dependencies ---
echo "📚 Installing dependencies..."
if [ -f "wa_analyzer/requirements.txt" ]; then
    $PYTHON_CMD -m pip install -r wa_analyzer/requirements.txt --quiet
else
    echo "⚠️  requirements.txt not found, installing essential packages..."
    $PYTHON_CMD -m pip install streamlit pandas plotly matplotlib seaborn wordcloud emoji gender-guesser vobject attrs --quiet
fi

# --- 6. Run the Streamlit app ---
echo ""
echo "============================================"
echo "✨ Setup complete! Launching the app..."
echo "============================================"
echo ""
$PYTHON_CMD -m streamlit run app.py
