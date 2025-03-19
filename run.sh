#!/bin/bash
# filepath: cv-demo/run.sh

# Run main.py with predefined parameters
uv run main.py \
    --cam_source 0 \
    --pose_model large \
    --detect_model large \
    --segment_model large \
    --device cuda