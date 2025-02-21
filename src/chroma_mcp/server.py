from typing import Dict, List, Optional
import chromadb
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import argparse
from chromadb.config import Settings


# Initialize FastMCP server
mcp = FastMCP("chroma")

# Global variables
_chroma_client = None

def create_parser():
    """Create and return the argument parser."""
    parser = argparse.ArgumentParser(description='FastMCP server for Chroma DB')
    parser.add_argument('--client-type', 
                       choices=['http', 'cloud', 'persistent', 'ephemeral'],
                       default=os.getenv('CHROMA_CLIENT_TYPE', 'ephemeral'),
                       help='Type of Chroma client to use')
    parser.add_argument('--data-dir',
                       default=os.getenv('CHROMA_DATA_DIR'),
                       help='Directory for persistent client data (only used with persistent client)')
    parser.add_argument('--host', 
                       help='Chroma host (required for http client)', 
                       default=os.getenv('CHROMA_HOST'))
    parser.add_argument('--port', 
                       help='Chroma port (optional for http client)', 
                       default=os.getenv('CHROMA_PORT'))
    parser.add_argument('--custom-auth-credentials',
                       help='Custom auth credentials (optional for http client)', 
                       default=os.getenv('CHROMA_CUSTOM_AUTH_CREDENTIALS'))
    parser.add_argument('--tenant', 
                       help='Chroma tenant (optional for http client)', 
                       default=os.getenv('CHROMA_TENANT'))
    parser.add_argument('--database', 
                       help='Chroma database (required if tenant is provided)', 
                       default=os.getenv('CHROMA_DATABASE'))
    parser.add_argument('--api-key', 
                       help='Chroma API key (required if tenant is provided)', 
                       default=os.getenv('CHROMA_API_KEY'))
    parser.add_argument('--ssl', 
                       help='Use SSL (optional for http client)', 
                       type=bool, 
                       default=os.getenv('CHROMA_SSL', 'true').lower() == 'true')
    parser.add_argument('--dotenv-path', 
                       help='Path to .chroma_env file', 
                       default=os.getenv('CHROMA_DOTENV_PATH', '.chroma_env'))
    return parser

def get_chroma_client(args=None):
    """Get or create the global Chroma client instance."""
    global _chroma_client
    if _chroma_client is None:
        if args is None:
            # Create parser and parse args if not provided
            parser = create_parser()
            args = parser.parse_args()
        
        # Load environment variables from .env file if it exists
        load_dotenv(dotenv_path=args.dotenv_path)
        
        if args.client_type == 'http':
            if not args.host:
                raise ValueError("Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client")
            
            settings = Settings()
            if args.custom_auth_credentials:
                settings = Settings(
                    chroma_client_auth_provider="chromadb.auth.basic_authn.BasicAuthClientProvider",
                    chroma_client_auth_credentials=args.custom_auth_credentials
                )
            
            _chroma_client = chromadb.HttpClient(
                host=args.host,
                port=args.port if args.port else None,
                ssl=args.ssl,
                settings=settings
            )
            
        elif args.client_type == 'cloud':
            if not args.tenant:
                raise ValueError("Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client")
            if not args.database:
                raise ValueError("Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client")
            if not args.api_key:
                raise ValueError("API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client")
            
            _chroma_client = chromadb.HttpClient(
                host="api.trychroma.com",
                ssl=True,  # Always use SSL for cloud
                tenant=args.tenant,
                database=args.database,
                headers={
                    'x-chroma-token': args.api_key
                }
            )
                
        elif args.client_type == 'persistent':
            if not args.data_dir:
                raise ValueError("Data directory must be provided via --data-dir flag when using persistent client")
            _chroma_client = chromadb.PersistentClient(path=args.data_dir)
        else:  # ephemeral
            _chroma_client = chromadb.EphemeralClient()
            
    return _chroma_client

##### Collection Tools #####

@mcp.tool()
async def list_collections(
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[str]:
    """List all collection names in the Chroma database with pagination support.
    
    Args:
        limit: Optional maximum number of collections to return
        offset: Optional number of collections to skip before returning results
    
    Returns:
        List of collection names
    """
    client = get_chroma_client()
    return client.list_collections(limit=limit, offset=offset)


@mcp.tool()
async def create_collection(
    collection_name: str,
    hnsw_space: Optional[str] = None,
    hnsw_construction_ef: Optional[int] = None,
    hnsw_search_ef: Optional[int] = None,
    hnsw_M: Optional[int] = None,
    hnsw_num_threads: Optional[int] = None,
    hnsw_resize_factor: Optional[float] = None,
    hnsw_batch_size: Optional[int] = None,
    hnsw_sync_threshold: Optional[int] = None
) -> str:
    """Create a new Chroma collection with configurable HNSW parameters.
    
    Args:
        collection_name: Name of the collection to create
        hnsw_space: Distance function used in HNSW index. Options: 'l2', 'ip', 'cosine'
        hnsw_construction_ef: Size of the dynamic candidate list for constructing the HNSW graph
        hnsw_search_ef: Size of the dynamic candidate list for searching the HNSW graph
        hnsw_M: Number of bi-directional links created for every new element
        hnsw_num_threads: Number of threads to use during HNSW construction
        hnsw_resize_factor: Factor to resize the index by when it's full
        hnsw_batch_size: Number of elements to batch together during index construction
        hnsw_sync_threshold: Number of elements to process before syncing index to disk
    """
    client = get_chroma_client()
    
    # Build HNSW configuration directly in metadata, only including non-None values
    metadata = {
        k: v for k, v in {
            "hnsw:space": hnsw_space,
            "hnsw:construction_ef": hnsw_construction_ef,
            "hnsw:M": hnsw_M,
            "hnsw:search_ef": hnsw_search_ef,
            "hnsw:num_threads": hnsw_num_threads,
            "hnsw:resize_factor": hnsw_resize_factor,
            "hnsw:batch_size": hnsw_batch_size,
            "hnsw:sync_threshold": hnsw_sync_threshold
        }.items() if v is not None
    }
    
    client.create_collection(
        name=collection_name,
        metadata=metadata if metadata else None
    )
    
    config_msg = f" with HNSW configuration: {metadata}" if metadata else ""
    return f"Successfully created collection {collection_name}{config_msg}"

@mcp.tool()
async def peek_collection(
    collection_name: str,
    limit: int = 5
) -> Dict:
    """Peek at documents in a Chroma collection.
    
    Args:
        collection_name: Name of the collection to peek into
        limit: Number of documents to peek at
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    results = collection.peek(limit=limit)
    return results

@mcp.tool()
async def get_collection_info(collection_name: str) -> Dict:
    """Get information about a Chroma collection.
    
    Args:
        collection_name: Name of the collection to get info about
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    
    # Get collection count
    count = collection.count()
    
    # Peek at a few documents
    peek_results = collection.peek(limit=3)
    
    return {
        "name": collection_name,
        "count": count,
        "sample_documents": peek_results
    }
    
@mcp.tool()
async def get_collection_count(collection_name: str) -> int:
    """Get the number of documents in a Chroma collection.
    
    Args:
        collection_name: Name of the collection to count
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    return collection.count()

@mcp.tool()
async def modify_collection(
    collection_name: str,
    new_name: Optional[str] = None,
    new_metadata: Optional[Dict] = None
) -> str:
    """Modify a Chroma collection's name or metadata.
    
    Args:
        collection_name: Name of the collection to modify
        new_name: Optional new name for the collection
        new_metadata: Optional new metadata for the collection
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    
    if new_name:
        collection.modify(name=new_name)
    if new_metadata:
        collection.modify(metadata=new_metadata)
    
    modified_aspects = []
    if new_name:
        modified_aspects.append("name")
    if new_metadata:
        modified_aspects.append("metadata")
    
    return f"Successfully modified collection {collection_name}: updated {' and '.join(modified_aspects)}"

    
@mcp.tool()
async def delete_collection(collection_name: str) -> str:
    """Delete a Chroma collection.
    
    Args:
        collection_name: Name of the collection to delete
    """
    client = get_chroma_client()
    client.delete_collection(collection_name)
    return f"Successfully deleted collection {collection_name}"

##### Document Tools #####
@mcp.tool()
async def add_documents(
    collection_name: str,
    documents: List[str],
    metadatas: Optional[List[Dict]] = None,
    ids: Optional[List[str]] = None
) -> str:
    """Add documents to a Chroma collection.
    
    Args:
        collection_name: Name of the collection to add documents to
        documents: List of text documents to add
        metadatas: Optional list of metadata dictionaries for each document
        ids: Optional list of IDs for the documents
    """
    client = get_chroma_client()
    collection = client.get_or_create_collection(collection_name)
    
    # Generate sequential IDs if none provided
    if ids is None:
        ids = [str(i) for i in range(len(documents))]
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids
    )
    
    return f"Successfully added {len(documents)} documents to collection {collection_name}"

@mcp.tool()
async def query_documents(
    collection_name: str,
    query_texts: List[str],
    n_results: int = 5,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None
) -> Dict:
    """Query documents from a Chroma collection with advanced filtering.
    
    Args:
        collection_name: Name of the collection to query
        query_texts: List of query texts to search for
        n_results: Number of results to return per query
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: Optional list of what to include in response. Can contain any of:
                ["documents", "embeddings", "metadatas", "distances"]
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    
    return collection.query(
        query_texts=query_texts,
        n_results=n_results,
        where=where,
        where_document=where_document,
        include=include
    )

@mcp.tool()
async def get_documents(
    collection_name: str,
    ids: Optional[List[str]] = None,
    where: Optional[Dict] = None,
    where_document: Optional[Dict] = None,
    include: Optional[List[str]] = None,
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict:
    """Get documents from a Chroma collection with optional filtering.
    
    Args:
        collection_name: Name of the collection to get documents from
        ids: Optional list of document IDs to retrieve
        where: Optional metadata filters using Chroma's query operators
               Examples:
               - Simple equality: {"metadata_field": "value"}
               - Comparison: {"metadata_field": {"$gt": 5}}
               - Logical AND: {"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}
               - Logical OR: {"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}
        where_document: Optional document content filters
        include: Optional list of what to include in response. Can contain any of:
                ["documents", "embeddings", "metadatas"]
        limit: Optional maximum number of documents to return
        offset: Optional number of documents to skip before returning results
    
    Returns:
        Dictionary containing the matching documents, their IDs, and requested includes
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    
    return collection.get(
        ids=ids,
        where=where,
        where_document=where_document,
        include=include,
        limit=limit,
        offset=offset
    )

def main():
    """Entry point for the Chroma MCP server."""
    parser = create_parser()
    args = parser.parse_args()
    
    # Validate required arguments based on client type
    if args.client_type == 'http':
        if not args.host:
            parser.error("Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client")
    
    elif args.client_type == 'cloud':
        if not args.tenant:
            parser.error("Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client")
        if not args.database:
            parser.error("Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client")
        if not args.api_key:
            parser.error("API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client")
    
    # Initialize client with parsed args
    get_chroma_client(args)
    
    # Initialize and run the server
    mcp.run(transport='stdio')
    
if __name__ == "__main__":
    main()
