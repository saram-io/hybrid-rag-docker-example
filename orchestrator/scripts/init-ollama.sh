#!/bin/bash
set -e

# Wait for Ollama API
MAX_RETRIES=60
RETRY_COUNT=0
while ! ollama list > /dev/null 2>&1; do
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "ERROR: Ollama API not ready"
        exit 1
    fi
    sleep 5
done

# Pull embedding model
if ! ollama list | grep -q "^qwen3-embedding:8b"; then
    echo "Pulling qwen3-embedding:8b..."
    ollama pull qwen3-embedding:8b
fi

# Pull frontier model
if ! ollama list | grep -q "^qwen3:32b"; then
    echo "Pulling qwen3:32b..."
    ollama pull qwen3:32b
fi

echo "Models ready."
ollama list
