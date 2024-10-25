#!/bin/bash

# Script name: run_pipeline.sh

# Set session name
SESSION_NAME="pipeline_experiment"

# Set your conda environment name here
CONDA_ENV="mip"  # Replace with your environment name

# Create log directory if it doesn't exist
mkdir -p tmux_logs

# Get current timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Check if the session exists
tmux has-session -t $SESSION_NAME 2>/dev/null

if [ $? != 0 ]; then
    # Create a new session
    tmux new-session -d -s $SESSION_NAME
    
    # Set up window
    tmux rename-window -t $SESSION_NAME:0 'pipeline'
    
    # Configure logging
    tmux pipe-pane -t $SESSION_NAME:0 "cat >> tmux_logs/pipeline_${TIMESTAMP}.log"
    
    # Initialize conda for bash shell
    tmux send-keys -t $SESSION_NAME:0 'eval "$(conda shell.bash hook)"' C-m
    
    # Activate conda environment
    tmux send-keys -t $SESSION_NAME:0 "conda activate $CONDA_ENV" C-m
    
    # Navigate to the correct directory (modify path as needed)
    tmux send-keys -t $SESSION_NAME:0 "cd $(pwd)" C-m
    
    # Run the pipeline with nohup to ensure it continues running
    tmux send-keys -t $SESSION_NAME:0 "python pipeline.py > pipeline_output_${TIMESTAMP}.log 2>&1" C-m
    
    echo "Started pipeline in tmux session: $SESSION_NAME"
    echo "Logging to: tmux_logs/pipeline_${TIMESTAMP}.log"
    echo "Additional output in: pipeline_output_${TIMESTAMP}.log"
else
    echo "Session $SESSION_NAME already exists"
    echo "Use 'tmux attach -t $SESSION_NAME' to connect to it"
fi