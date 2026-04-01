#!/bin/bash
# Start both backend and frontend dev servers.
# Run from the abstraction/ project root.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Start FastAPI backend
echo "Starting FastAPI backend on :8000..."
cd "$PROJECT_DIR"
uvicorn abstraction.app:app --reload --port 8000 &
BACKEND_PID=$!

# Start SvelteKit frontend
echo "Starting SvelteKit frontend on :5173..."
cd "$PROJECT_DIR/frontend"
npm run dev &
FRONTEND_PID=$!

# Cleanup on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT

echo ""
echo "Backend:  http://localhost:8000/docs"
echo "Frontend: http://localhost:5173"
echo ""
echo "Press Ctrl+C to stop both servers."

wait
