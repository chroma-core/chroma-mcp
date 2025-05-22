# Chroma MCP Server for Docker

This implementation provides a properly configured Chroma MCP server designed to run in a Docker container, exposing HTTP endpoints for Claude to interact with the Chroma vector database.

## 📋 Table of Contents

- [Overview](#overview)
- [Implementation](#implementation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Troubleshooting](#troubleshooting)

## Overview

This MCP server implementation connects to a Chroma vector database instance and exposes the Chroma functionality as MCP tools that Claude can use. It is specifically designed to work in a Docker container environment with proper network connectivity between services.

## Implementation

The implementation consists of several key components:

1. **Dockerfile**: A custom Dockerfile that installs the necessary dependencies and configures the container environment.

2. **Entrypoint Script**: A Bash script that starts the MCP server with the appropriate configuration.

3. **Server Script**: A Python script that sets up the FastMCP instance with HTTP transport.

4. **Docker Compose Configuration**: A configuration that connects the Chroma MCP server to the Chroma database.

## Configuration

The server is highly configurable through environment variables:

### Chroma Connection

- `CHROMA_CLIENT_TYPE`: Type of Chroma client to use (`http`, `cloud`, `persistent`, `ephemeral`)
- `CHROMA_HOST`: Hostname of the Chroma server
- `CHROMA_PORT`: Port of the Chroma server
- `CHROMA_SSL`: Whether to use SSL for Chroma connections (`true`/`false`)
- `CHROMA_DATA_DIR`: Directory for persistent data (only for persistent client type)

### MCP Server

- `MCP_HOST`: Host to bind the MCP server to
- `MCP_PORT`: Port to expose the MCP server on
- `MCP_TRANSPORT`: Transport mechanism for the MCP server (`http` or `stdio`)

## Usage

### With Docker Compose

1. Use the provided `docker-compose.yml` file to start both Chroma and the Chroma MCP server:

```bash
docker-compose up -d
```

2. Configure Claude to use the MCP server:

```bash
claude mcp add chroma http://localhost:8080
```

### Standalone Docker

You can also run the container standalone:

```bash
docker build -t chroma-mcp -f Dockerfile.new .
docker run -p 8080:8080 \
  -e CHROMA_CLIENT_TYPE=http \
  -e CHROMA_HOST=hostname \
  -e CHROMA_PORT=8000 \
  -e CHROMA_SSL=false \
  chroma-mcp
```

## Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure the Chroma database is running and accessible from the MCP container.

2. **MCP Server Not Starting**: Check the logs for any startup errors:

```bash
docker logs chroma-mcp
```

3. **Claude Can't Connect**: Verify that the HTTP endpoint is accessible and properly configured in Claude:

```bash
curl http://localhost:8080/health
```

---

🧭 [Home](../../../README.md) | [Up](../README.md)

*Last updated: May 20, 2025*