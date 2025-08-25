#!/bin/bash

# Script to run automator tests in Docker

set -e

echo "Building automator Docker image..."
docker build -t automator:latest .

echo "Running tests in Docker container..."
docker run --rm \
  -e OPENAI_API_KEY="${OPENAI_API_KEY}" \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e GEMINI_API_KEY="${GEMINI_API_KEY}" \
  -e HF_TOKEN="${HF_TOKEN}" \
  -e OPENWEIGHTS_API_KEY="${OPENWEIGHTS_API_KEY}" \
  -e SERP_API_KEY="${SERP_API_KEY}" \
  automator:latest

echo "Tests completed!"