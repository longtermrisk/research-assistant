#!/bin/bash
set -eo pipefail

echo "===== Docker Entrypoint: Starting Automator Setup ====="

# Step 1: Generate mcp.json from environment variables
echo "--> Running MCP configuration script..."
python3 /app/configure_mcp.py
if [ $? -ne 0 ]; then
    echo "Error: MCP configuration failed. Exiting."
    exit 1
fi
echo "MCP configuration complete."
echo

# Step 2: Run the test suite to verify the installation
echo "--> Running test suite..."
cd / && PYTHONPATH="" python3 /app/docker_test.py
if [ $? -ne 0 ]; then
    echo "Error: Test suite failed. Please check the logs."
    echo "Exiting without starting services."
    exit 1
fi
echo "All tests passed successfully!"
echo

# Step 3: Start the main application services
echo "--> Starting Automator services..."

# Start backend server in the background
echo "Starting backend on 0.0.0.0:8000..."
cd /app/automator
uvicorn automator.api.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Start frontend server in the background
echo "Starting frontend on 0.0.0.0:5173..."
cd /app/automator/ui/frontend
npm run dev -- --host 0.0.0.0 --port 5173 &
FRONTEND_PID=$!

echo
echo "====================================================="
echo " Automator is running!"
echo " - Backend API: http://localhost:8000"
echo " - Frontend UI: http://localhost:5173"
echo "====================================================="
echo

# Wait for background processes to exit
# This will keep the container running
wait $BACKEND_PID $FRONTEND_PID
EXIT_CODE=$?

echo "One of the services has stopped. Exiting container."
exit $EXIT_CODE