FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends gcc

# Install necessary Python libraries first
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir fastapi uvicorn fastmcp chromadb

# Copy application files
COPY ./src /app/src
COPY ./pyproject.toml /app/
COPY ./entrypoint.sh /app/
COPY ./server.py /app/

# Install the package in development mode
RUN pip install -e .

# Make the entrypoint script executable
RUN chmod +x /app/entrypoint.sh

# Expose the port for the MCP server
EXPOSE 8080

# Set environment variables with defaults (can be overridden at runtime)
ENV CHROMA_CLIENT_TYPE=http
ENV CHROMA_HOST=chroma
ENV CHROMA_PORT=8000
ENV CHROMA_SSL=false
ENV MCP_PORT=8080
ENV MCP_HOST=0.0.0.0

# Run the entrypoint script
ENTRYPOINT ["/app/entrypoint.sh"]