#!/bin/bash
set -e

# Print environment variables for debugging
echo "Starting Chroma MCP server with the following configuration:"
echo "CHROMA_CLIENT_TYPE: $CHROMA_CLIENT_TYPE"
echo "CHROMA_HOST: $CHROMA_HOST"
echo "CHROMA_PORT: $CHROMA_PORT"
echo "CHROMA_SSL: $CHROMA_SSL"
echo "MCP_PORT: $MCP_PORT"
echo "MCP_HOST: $MCP_HOST"

# Check if server.py is accessible
if [ ! -f "/app/server.py" ]; then
    echo "Error: server.py not found in /app directory"
    ls -la /app
    exit 1
fi

# Run the server
echo "Starting MCP server with HTTP transport..."
exec python /app/server.py \
    --client-type "$CHROMA_CLIENT_TYPE" \
    --host "$CHROMA_HOST" \
    --port "$CHROMA_PORT" \
    --ssl "$CHROMA_SSL" \
    --transport http \
    --mcp-host "$MCP_HOST" \
    --mcp-port "$MCP_PORT"