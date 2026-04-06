#!/bin/bash

echo "Starting Finance Risk Assessment Application..."

# Start FastAPI server
echo "Starting FastAPI server..."
uvicorn app.main:app --reload --port 8000 &

# Start Streamlit frontend  
sleep 3
echo "Starting Streamlit frontend..."
streamlit run frontend/streamlit_app.py

wait
