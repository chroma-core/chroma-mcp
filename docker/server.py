#!/usr/bin/env python3
"""
HTTP server wrapper for Chroma MCP.
This script starts a Chroma MCP server with HTTP transport.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
import importlib.util

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chroma-mcp")

def get_parser():
    parser = argparse.ArgumentParser(description='Start Chroma MCP server with HTTP transport')
    
    # Chroma connection parameters
    parser.add_argument('--client-type', 
                      choices=['http', 'cloud', 'persistent', 'ephemeral'],
                      default=os.getenv('CHROMA_CLIENT_TYPE', 'http'),
                      help='Type of Chroma client to use')
    parser.add_argument('--host', 
                      help='Chroma host (required for http client)', 
                      default=os.getenv('CHROMA_HOST', 'localhost'))
    parser.add_argument('--port', 
                      help='Chroma port (optional for http client)', 
                      default=os.getenv('CHROMA_PORT', '8000'))
    parser.add_argument('--ssl', 
                      help='Use SSL (optional for http client)', 
                      type=lambda x: x.lower() in ['true', 'yes', '1', 't', 'y'],
                      default=os.getenv('CHROMA_SSL', 'false').lower() in ['true', 'yes', '1', 't', 'y'])
    
    # MCP Server parameters
    parser.add_argument('--transport', 
                      choices=['stdio', 'http'],
                      default=os.getenv('MCP_TRANSPORT', 'http'),
                      help='Transport to use for MCP server')
    parser.add_argument('--mcp-host', 
                      help='Host to bind MCP server to',
                      default=os.getenv('MCP_HOST', '0.0.0.0'))
    parser.add_argument('--mcp-port', 
                      type=int,
                      help='Port to bind MCP server to',
                      default=int(os.getenv('MCP_PORT', '8080')))
    
    # Advanced parameters
    parser.add_argument('--data-dir',
                      default=os.getenv('CHROMA_DATA_DIR'),
                      help='Directory for persistent client data (only used with persistent client)')
    parser.add_argument('--dotenv-path', 
                      help='Path to .env file', 
                      default=os.getenv('CHROMA_DOTENV_PATH', '.chroma_env'))
    parser.add_argument('--debug', 
                      action='store_true', 
                      help='Enable debug logging')
    
    return parser

async def start_mcp_server():
    """Start the MCP server with the configured transport."""
    parser = get_parser()
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Import the Chroma MCP module
    logger.info("Importing Chroma MCP module")
    try:
        # Try to import from source directory first
        src_path = Path(__file__).parent / "src" / "chroma_mcp" / "server.py"
        if src_path.exists():
            logger.info(f"Loading from source path: {src_path}")
            spec = importlib.util.spec_from_file_location("chroma_mcp", src_path)
            server_module = importlib.util.module_from_spec(spec)
            sys.modules["chroma_mcp"] = server_module
            spec.loader.exec_module(server_module)
        else:
            # Otherwise import from installed package
            logger.info("Loading from installed package")
            import chroma_mcp.server as server_module
        
        # Get the FastMCP instance and initialize the client
        mcp = server_module.mcp
        
        # Prepare args for the client initialization
        client_args = [
            "--client-type", args.client_type,
            "--host", args.host,
            "--port", args.port,
            "--ssl", "true" if args.ssl else "false"
        ]
        
        if args.data_dir:
            client_args.extend(["--data-dir", args.data_dir])
        
        logger.info(f"Initializing Chroma client with args: {client_args}")
        client_parser = server_module.create_parser()
        client_args = client_parser.parse_args(client_args)
        
        try:
            server_module.get_chroma_client(client_args)
            logger.info("Successfully initialized Chroma client")
        except Exception as e:
            logger.error(f"Failed to initialize Chroma client: {str(e)}")
            sys.exit(1)
        
        # Run the server with the specified transport
        if args.transport == 'http':
            logger.info(f"Starting MCP server with HTTP transport on {args.mcp_host}:{args.mcp_port}")
            return await mcp.run_server(host=args.mcp_host, port=args.mcp_port)
        else:
            logger.info("Starting MCP server with stdio transport")
            return mcp.run(transport='stdio')
    
    except Exception as e:
        logger.error(f"Error starting MCP server: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    import asyncio
    asyncio.run(start_mcp_server())