#!/bin/bash

echo "Running Python Server Tests"
pytest tests/testserver.py || exit 1

echo "Running Unity WebSocket Client Tests"
/path/to/Unity -batchmode -projectPath /path/to/Project -runTests -testPlatform playmode -logFile || exit 1

echo "All tests passed"
