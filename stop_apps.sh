#!/bin/bash

# Stop Script - Kills both Flask and Ionic servers
# Usage: ./stop_apps.sh

echo "Stopping Brain Tumor App servers..."
echo ""

# Kill Flask servers
echo "Stopping Flask servers..."
pkill -f "flask run" 2>/dev/null
pkill -f "python.*flask" 2>/dev/null
sleep 1

# Kill Ionic servers
echo "Stopping Ionic servers..."
pkill -f "ionic serve" 2>/dev/null
sleep 1

# Kill any remaining node processes from Ionic
pkill -f "ng serve" 2>/dev/null

echo ""
echo "✅ All servers stopped"
echo ""
