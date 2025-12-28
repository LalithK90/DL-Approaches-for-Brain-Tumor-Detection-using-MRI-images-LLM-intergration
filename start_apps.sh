#!/bin/bash

# Brain Tumor Identification App - Development Server Startup Script
# This script starts both Flask backend and Ionic frontend applications
# Location: brain_tumor_app/start_apps.sh

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get the base directory (where this script is located)
BASE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Brain Tumor Identification App${NC}"
echo -e "${BLUE}Starting Development Servers...${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# ACTIVATE CONDA ENVIRONMENT FIRST
echo -e "${YELLOW}🐍 Activating conda environment 'mri_data'...${NC}"

# Initialize conda for bash - check multiple possible locations
if [ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/homebrew/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/opt/homebrew/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/usr/local/anaconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/anaconda3/etc/profile.d/conda.sh"
elif [ -f "/usr/local/miniconda3/etc/profile.d/conda.sh" ]; then
    source "/usr/local/miniconda3/etc/profile.d/conda.sh"
else
    echo -e "${RED}❌ Error: Could not find conda installation${NC}"
    echo -e "${YELLOW}Please ensure conda is installed and accessible${NC}"
    echo -e "${YELLOW}Detected conda at: $(which conda 2>/dev/null || echo 'not found')${NC}"
    exit 1
fi

# Activate the mri_data environment
conda activate mri_data

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Failed to activate conda environment 'mri_data'${NC}"
    echo -e "${YELLOW}Please create the environment or activate it manually${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Conda environment 'mri_data' activated${NC}"
echo ""

# Define directories
FLASK_DIR="$BASE_DIR/brain_tumor_identification_api"
IONIC_DIR="$BASE_DIR/braintumoridentificationapp"

# Check if directories exist
if [ ! -d "$FLASK_DIR" ]; then
    echo -e "${RED}❌ Error: Flask directory not found at $FLASK_DIR${NC}"
    exit 1
fi

if [ ! -d "$IONIC_DIR" ]; then
    echo -e "${RED}❌ Error: Ionic directory not found at $IONIC_DIR${NC}"
    exit 1
fi

# Create logs directory
LOGS_DIR="$BASE_DIR/logs"
mkdir -p "$LOGS_DIR"

# Log file paths
FLASK_LOG="$LOGS_DIR/flask_server.log"
IONIC_LOG="$LOGS_DIR/ionic_server.log"

echo -e "${YELLOW}📁 Base Directory: $BASE_DIR${NC}"
echo -e "${YELLOW}🔧 Flask Directory: $FLASK_DIR${NC}"
echo -e "${YELLOW}⚛️  Ionic Directory: $IONIC_DIR${NC}"
echo -e "${YELLOW}📝 Logs Directory: $LOGS_DIR${NC}"
echo ""

# Function to handle script termination
cleanup() {
    echo ""
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Stopping servers...${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Kill Flask process
    if [ ! -z "$FLASK_PID" ]; then
        echo -e "${YELLOW}Stopping Flask server (PID: $FLASK_PID)...${NC}"
        kill $FLASK_PID 2>/dev/null
        wait $FLASK_PID 2>/dev/null
    fi
    
    # Kill Ionic process
    if [ ! -z "$IONIC_PID" ]; then
        echo -e "${YELLOW}Stopping Ionic server (PID: $IONIC_PID)...${NC}"
        kill $IONIC_PID 2>/dev/null
        wait $IONIC_PID 2>/dev/null
    fi
    
    echo -e "${GREEN}✅ All servers stopped${NC}"
    exit 0
}

# Set trap to handle Ctrl+C
trap cleanup SIGINT SIGTERM

# Start Flask Server
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Starting Flask Backend Server...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$FLASK_DIR"

# Check if Flask is installed
if ! python -c "import flask" 2>/dev/null; then
    echo -e "${RED}❌ Flask is not installed${NC}"
    echo -e "${YELLOW}Please install requirements: pip install -r requirements.txt${NC}"
    exit 1
fi

# Start Flask in background
echo -e "${GREEN}▶️  Starting Flask on http://localhost:5001${NC}"
python app.py > "$FLASK_LOG" 2>&1 &
FLASK_PID=$!

# Give Flask time to start
sleep 3

# Check if Flask started successfully
if ! kill -0 $FLASK_PID 2>/dev/null; then
    echo -e "${RED}❌ Flask failed to start${NC}"
    echo -e "${YELLOW}Check logs: cat $FLASK_LOG${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Flask server started (PID: $FLASK_PID)${NC}"
echo -e "${YELLOW}📝 Log file: $FLASK_LOG${NC}"
echo ""

# Start Ionic Server
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}Starting Ionic Frontend Server...${NC}"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

cd "$IONIC_DIR"

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo -e "${RED}❌ Node.js is not installed${NC}"
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo -e "${RED}❌ npm is not installed${NC}"
    exit 1
fi

# We use 'npm run dev' which doesn't require ionic CLI

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo -e "${YELLOW}⚠️  node_modules not found${NC}"
    echo -e "${YELLOW}Installing dependencies...${NC}"
    npm install
fi

# Start Vite dev server in background (replaces ionic serve with proper proxy support)
echo -e "${GREEN}▶️  Starting Frontend on http://localhost:8100${NC}"
npm run dev > "$IONIC_LOG" 2>&1 &
IONIC_PID=$!

# Give Ionic time to start
sleep 5

# Check if Ionic started successfully
if ! kill -0 $IONIC_PID 2>/dev/null; then
    echo -e "${RED}❌ Ionic failed to start${NC}"
    echo -e "${YELLOW}Check logs: cat $IONIC_LOG${NC}"
    kill $FLASK_PID 2>/dev/null
    exit 1
fi

echo -e "${GREEN}✅ Frontend server started (PID: $IONIC_PID)${NC}"
echo -e "${YELLOW}📝 Log file: $IONIC_LOG${NC}"
echo ""

# Display server information
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}✅ All servers running successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}🔗 Access the application:${NC}"
echo -e "${GREEN}   Frontend: http://localhost:5173${NC}"
echo -e "${GREEN}   Backend:  http://localhost:5001${NC}"
echo ""
echo -e "${YELLOW}📊 Process Information:${NC}"
echo -e "   Flask (Backend):  PID $FLASK_PID"
echo -e "   Ionic (Frontend): PID $IONIC_PID"
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo -e "   Flask Log:  $FLASK_LOG"
echo -e "   Ionic Log:  $IONIC_LOG"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop all servers${NC}"
echo ""

# Wait for both processes to continue running
wait
