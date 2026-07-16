from typing import Dict, List, Optional, TypedDict, Union
from enum import Enum
import chromadb
from mcp.server.fastmcp import FastMCP
import os
from dotenv import load_dotenv
import argparse
from chromadb.config import Settings
import ssl
import uuid
import time
import json
from typing_extensions import TypedDict


from chromadb.api.collection_configuration import (
    CreateCollectionConfiguration
    )
from chromadb.api import EmbeddingFunction
from chromadb.utils.embedding_functions import (
    DefaultEmbeddingFunction,
    CohereEmbeddingFunction,
    OpenAIEmbeddingFunction,
    JinaEmbeddingFunction,
    VoyageAIEmbeddingFunction,
    RoboflowEmbeddingFunction,
)

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
                       type=lambda x: x.lower() in ['true', 'yes', '1', 't', 'y'],
                       default=os.getenv('CHROMA_SSL', 'true').lower() in ['true', 'yes', '1', 't', 'y'])
    parser.add_argument('--dotenv-path', 
                       help='Path to .env file', 
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
            
            # Handle SSL configuration
            try:
                _chroma_client = chromadb.HttpClient(
                    host=args.host,
                    port=args.port if args.port else None,
                    ssl=args.ssl,
                    settings=settings
                )
            except ssl.SSLError as e:
                print(f"SSL connection failed: {str(e)}")
                raise
            except Exception as e:
                print(f"Error connecting to HTTP client: {str(e)}")
                raise
            
        elif args.client_type == 'cloud':
            if not args.tenant:
                raise ValueError("Tenant must be provided via --tenant flag or CHROMA_TENANT environment variable when using cloud client")
            if not args.database:
                raise ValueError("Database must be provided via --database flag or CHROMA_DATABASE environment variable when using cloud client")
            if not args.api_key:
                raise ValueError("API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client")
            
            try:
                _chroma_client = chromadb.HttpClient(
                    host="api.trychroma.com",
                    ssl=True,  # Always use SSL for cloud
                    tenant=args.tenant,
                    database=args.database,
                    headers={
                        'x-chroma-token': args.api_key
                    }
                )
            except ssl.SSLError as e:
                print(f"SSL connection failed: {str(e)}")
                raise
            except Exception as e:
                print(f"Error connecting to cloud client: {str(e)}")
                raise
                
        elif args.client_type == 'persistent':
            if not args.data_dir:
                raise ValueError("Data directory must be provided via --data-dir flag when using persistent client")
            _chroma_client = chromadb.PersistentClient(path=args.data_dir)
        else:  # ephemeral
            _chroma_client = chromadb.EphemeralClient()
            
    return _chroma_client

##### Parse Tools #####

def _parse_dict_param(value: str | Dict | None, param_name: str) -> Dict | None:
    """Parse a dict parameter that may come as a JSON string.
    
    Agent frameworks pass JSON objects as strings (e.g., '{"$contains": "Python"}'),
    so we accept string input and parse it into a dict. Dict input is also accepted
    for backward compatibility with direct Python callers.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"Parameter '{param_name}' must be a JSON object, "
                    f"got {type(parsed).__name__}: {value}"
                )
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Parameter '{param_name}' is not valid JSON: {value}. "
                f"Error: {str(e)}"
            ) from e
    raise ValueError(
        f"Parameter '{param_name}' must be a dict or JSON string, "
        f"got {type(value).__name__}"
    )
    

def _parse_list_dict_param(value: str | List[Dict] | None, param_name: str) -> List[Dict] | None:
    """Parse a list-of-dicts parameter that may come as a JSON string."""
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Parameter '{param_name}' must be a JSON array, "
                    f"got {type(parsed).__name__}: {value}"
                )
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Parameter '{param_name}' is not valid JSON: {value}. "
                f"Error: {str(e)}"
            ) from e
    raise ValueError(
        f"Parameter '{param_name}' must be a list or JSON string, "
        f"got {type(value).__name__}"
    )


def _parse_list_param(value: str | List | None, param_name: str) -> List | None:
    """Parse a list parameter that may come as a JSON string.

    Agent frameworks pass JSON arrays as strings (e.g., '["id1","id2"]'),
    so we accept string input and parse it into a list. List input is also
    accepted for backward compatibility with direct Python callers.
    """
    if value is None:
        return None
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        try:
            parsed = json.loads(value)
            if not isinstance(parsed, list):
                raise ValueError(
                    f"Parameter '{param_name}' must be a JSON array, "
                    f"got {type(parsed).__name__}: {value}"
                )
            return parsed
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Parameter '{param_name}' is not valid JSON: {value}. "
                f"Error: {str(e)}"
            ) from e
    raise ValueError(
        f"Parameter '{param_name}' must be a list or JSON string, "
        f"got {type(value).__name__}"
    )


##### Collection Tools #####

@mcp.tool()
async def chroma_list_collections(
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> List[str]:
    """List all collection names in the Chroma database with pagination support.
    
    Args:
        limit: Optional maximum number of collections to return
        offset: Optional number of collections to skip before returning results
    
    Returns:
        List of collection names or ["__NO_COLLECTIONS_FOUND__"] if database is empty
    """
    client = get_chroma_client()
    try:
        colls = client.list_collections(limit=limit, offset=offset)
        # Safe handling: If colls is None or empty, return a special marker
        if not colls:
            return ["__NO_COLLECTIONS_FOUND__"]
        # Otherwise iterate to get collection names
        return [coll.name for coll in colls]

    except Exception as e:
        raise Exception(f"Failed to list collections: {str(e)}") from e

mcp_known_embedding_functions: Dict[str, EmbeddingFunction] = {
    "default": DefaultEmbeddingFunction,
    "cohere": CohereEmbeddingFunction,
    "openai": OpenAIEmbeddingFunction,
    "jina": JinaEmbeddingFunction,
    "voyageai": VoyageAIEmbeddingFunction,
    "roboflow": RoboflowEmbeddingFunction,
}
@mcp.tool()
async def chroma_create_collection(
    collection_name: str,
    embedding_function_name: str = "default",
    metadata: Optional[Union[str, Dict]] = None,
) -> str:
    """Create a new Chroma collection with configurable HNSW parameters.
    
    Args:
        collection_name: Name of the collection to create
        embedding_function_name: Name of the embedding function to use. Options: 'default', 'cohere', 'openai', 'jina', 'voyageai', 'ollama', 'roboflow'
        metadata: Optional metadata dict as a JSON string or Dict (e.g., '{"key": "value"}' or {"key": "value"})
    """
    metadata = _parse_dict_param(metadata, "metadata")
    client = get_chroma_client()
    
    embedding_function = mcp_known_embedding_functions[embedding_function_name]
    
    configuration=CreateCollectionConfiguration(
        embedding_function=embedding_function()
    )
    
    try:
        client.create_collection(
            name=collection_name,
            configuration=configuration,
            metadata=metadata
        )
        config_msg = f" with configuration: {configuration}"
        return f"Successfully created collection {collection_name}{config_msg}"
    except Exception as e:
        raise Exception(f"Failed to create collection '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_peek_collection(
    collection_name: str,
    limit: int = 5
) -> Dict:
    """Peek at documents in a Chroma collection.
    
    Args:
        collection_name: Name of the collection to peek into
        limit: Number of documents to peek at
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        results = collection.peek(limit=limit)
        return results
    except Exception as e:
        raise Exception(f"Failed to peek collection '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_get_collection_info(collection_name: str) -> Dict:
    """Get information about a Chroma collection.
    
    Args:
        collection_name: Name of the collection to get info about
    """
    client = get_chroma_client()
    try:
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
    except Exception as e:
        raise Exception(f"Failed to get collection info for '{collection_name}': {str(e)}") from e
    
@mcp.tool()
async def chroma_get_collection_count(collection_name: str) -> int:
    """Get the number of documents in a Chroma collection.
    
    Args:
        collection_name: Name of the collection to count
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.count()
    except Exception as e:
        raise Exception(f"Failed to get collection count for '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_modify_collection(
    collection_name: str,
    new_name: Optional[str] = None,
    new_metadata: Optional[Union[str, Dict]] = None,
) -> str:
    """Modify a Chroma collection's name or metadata.
    
    Args:
        collection_name: Name of the collection to modify
        new_name: Optional new name for the collection
        new_metadata: Optional new metadata for the collection as a JSON string or Dict
    """
    new_metadata = _parse_dict_param(new_metadata, "new_metadata")
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        collection.modify(name=new_name, metadata=new_metadata)
        
        modified_aspects = []
        if new_name:
            modified_aspects.append("name")
        if new_metadata:
            modified_aspects.append("metadata")
        
        return f"Successfully modified collection {collection_name}: updated {' and '.join(modified_aspects)}"
    except Exception as e:
        raise Exception(f"Failed to modify collection '{collection_name}': {str(e)}") from e
    
@mcp.tool()
async def chroma_fork_collection(
    collection_name: str,
    new_collection_name: str,
) -> str:
    """Fork a Chroma collection.
    
    Args:
        collection_name: Name of the collection to fork
        new_collection_name: Name of the new collection to create
        metadata: Optional metadata dict to add to the new collection
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        collection.fork(new_collection_name)
        return f"Successfully forked collection {collection_name} to {new_collection_name}"
    except Exception as e:
        raise Exception(f"Failed to fork collection '{collection_name}': {str(e)}") from e
    
@mcp.tool()
async def chroma_delete_collection(collection_name: str) -> str:
    """Delete a Chroma collection.
    
    Args:
        collection_name: Name of the collection to delete
    """
    client = get_chroma_client()
    try:
        client.delete_collection(collection_name)
        return f"Successfully deleted collection {collection_name}"
    except Exception as e:
        raise Exception(f"Failed to delete collection '{collection_name}': {str(e)}") from e


@mcp.tool()
async def chroma_add_documents(
    collection_name: str,
    documents: Union[str, List[str]],
    ids: Union[str, List[str]],
    metadatas: Optional[Union[str, List[Dict]]] = None
) -> str:
    """Add documents to a Chroma collection.
    
    Args:
        collection_name: Name of the collection to add documents to
        documents: List of text documents to add, or a JSON string array (e.g., '["doc1","doc2"]')
        ids: List of IDs for the documents (required), or a JSON string array (e.g., '["id1","id2"]')
        metadatas: Optional list of metadata dictionaries as a JSON string or List[Dict]
                   (e.g., '[{"key": "value"}]' or [{"key": "value"}])
    """
    metadatas = _parse_list_dict_param(metadatas, "metadatas")
    documents = _parse_list_param(documents, "documents")
    ids = _parse_list_param(ids, "ids")
    if not documents:
        raise ValueError("The 'documents' list cannot be empty.")
    
    if not ids:
        raise ValueError("The 'ids' list is required and cannot be empty.")
    
    # Check if there are empty strings in the ids list
    if any(not id.strip() for id in ids):
        raise ValueError("IDs cannot be empty strings.")
    
    if len(ids) != len(documents):
        raise ValueError(f"Number of ids ({len(ids)}) must match number of documents ({len(documents)}).")

    client = get_chroma_client()
    try:
        collection = client.get_or_create_collection(collection_name)
        
        # Check for duplicate IDs
        existing_ids = collection.get(include=[])["ids"]
        duplicate_ids = [id for id in ids if id in existing_ids]
        
        if duplicate_ids:
            raise ValueError(
                f"The following IDs already exist in collection '{collection_name}': {duplicate_ids}. "
                f"Use 'chroma_update_documents' to update existing documents."
            )
        
        result = collection.add(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        
        # Check the return value
        if result and isinstance(result, dict):
            # If the return value is a dictionary, it may contain success information
            if 'success' in result and not result['success']:
                raise Exception(f"Failed to add documents: {result.get('error', 'Unknown error')}")
            
            # If the return value contains the actual number added
            if 'count' in result:
                return f"Successfully added {result['count']} documents to collection {collection_name}"
        
        # Default return
        return f"Successfully added {len(documents)} documents to collection {collection_name}, result is {result}"
    except Exception as e:
        raise Exception(f"Failed to add documents to collection '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_query_documents(
    collection_name: str,
    query_texts: Union[str, List[str]],
    n_results: int = 5,
    where: Optional[Union[str, Dict]] = None,
    where_document: Optional[Union[str, Dict]] = None,
    include: Union[str, List[str]] = "[\"documents\", \"metadatas\", \"distances\"]"
) -> Dict:
    """Query documents from a Chroma collection with advanced filtering.
    
    Args:
        collection_name: Name of the collection to query
        query_texts: List of query texts to search for, or a JSON string array (e.g., '["text1","text2"]')
        n_results: Number of results to return per query
        where: Optional metadata filters as a JSON string or Dict using Chroma's query operators
               Examples:
               - Simple equality: '{"metadata_field": "value"}' or {"metadata_field": "value"}
               - Comparison: '{"metadata_field": {"$gt": 5}}'
               - Logical AND: '{"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}'
               - Logical OR: '{"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}'
        where_document: Optional document content filters as a JSON string or Dict
               Examples:
               - Contains: '{"$contains": "value"}'
               - Not contains: '{"$not_contains": "value"}'
               - Regex: '{"$regex": "[a-z]+"}'
               - Not regex: '{"$not_regex": "[a-z]+"}'
               - Logical AND: '{"$and": [{"$contains": "value1"}, {"$not_regex": "[a-z]+"}]}'
               - Logical OR: '{"$or": [{"$regex": "[a-z]+"}, {"$not_contains": "value2"}]}'
        include: List of what to include in response. By default, this will include documents, metadatas, and distances.
    """
    if not query_texts:
        raise ValueError("The 'query_texts' list cannot be empty.")

    query_texts = _parse_list_param(query_texts, "query_texts")
    include = _parse_list_param(include, "include")
    where = _parse_dict_param(where, "where")
    where_document = _parse_dict_param(where_document, "where_document")

    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.query(
            query_texts=query_texts,
            n_results=n_results,
            where=where,
            where_document=where_document,
            include=include
        )
    except Exception as e:
        raise Exception(f"Failed to query documents from collection '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_get_documents(
    collection_name: str,
    ids: Optional[Union[str, List[str]]] = None,
    where: Optional[Union[str, Dict]] = None,
    where_document: Optional[Union[str, Dict]] = None,
    include: Union[str, List[str]] = "[\"documents\", \"metadatas\"]",
    limit: Optional[int] = None,
    offset: Optional[int] = None
) -> Dict:
    """Get documents from a Chroma collection with optional filtering.
    
    Args:
        collection_name: Name of the collection to get documents from
        ids: Optional list of document IDs to retrieve, or a JSON string array (e.g., '["id1","id2"]')
        where: Optional metadata filters as a JSON string or Dict using Chroma's query operators
               Examples:
               - Simple equality: '{"metadata_field": "value"}' or {"metadata_field": "value"}
               - Comparison: '{"metadata_field": {"$gt": 5}}'
               - Logical AND: '{"$and": [{"field1": {"$eq": "value1"}}, {"field2": {"$gt": 5}}]}'
               - Logical OR: '{"$or": [{"field1": {"$eq": "value1"}}, {"field1": {"$eq": "value2"}}]}'
        where_document: Optional document content filters as a JSON string or Dict
               Examples:
               - Contains: '{"$contains": "value"}'
               - Not contains: '{"$not_contains": "value"}'
               - Regex: '{"$regex": "[a-z]+"}'
               - Not regex: '{"$not_regex": "[a-z]+"}'
               - Logical AND: '{"$and": [{"$contains": "value1"}, {"$not_regex": "[a-z]+"}]}'
               - Logical OR: '{"$or": [{"$regex": "[a-z]+"}, {"$not_contains": "value2"}]}'
        include: List of what to include in response. By default, this will include documents, and metadatas.
        limit: Optional maximum number of documents to return
        offset: Optional number of documents to skip before returning results
    
    Returns:
        Dictionary containing the matching documents, their IDs, and requested includes
    """
    where = _parse_dict_param(where, "where")
    where_document = _parse_dict_param(where_document, "where_document")
    ids = _parse_list_param(ids, "ids")
    include = _parse_list_param(include, "include")
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
        return collection.get(
            ids=ids,
            where=where,
            where_document=where_document,
            include=include,
            limit=limit,
            offset=offset
        )
    except Exception as e:
        raise Exception(f"Failed to get documents from collection '{collection_name}': {str(e)}") from e

@mcp.tool()
async def chroma_update_documents(
    collection_name: str,
    ids: Union[str, List[str]],
    embeddings: Optional[Union[str, List[List[float]]]] = None,
    metadatas: Optional[Union[str, List[Dict]]] = None,
    documents: Optional[Union[str, List[str]]] = None
) -> str:
    """Update documents in a Chroma collection.

    Args:
        collection_name: Name of the collection to update documents in
        ids: List of document IDs to update (required), or a JSON string array (e.g., '["id1","id2"]')
        embeddings: Optional list of new embeddings for the documents.
                    Must match length of ids if provided.
        metadatas: Optional list of new metadata dictionaries as a JSON string or List[Dict]
                   (e.g., '[{"key": "value"}]' or [{"key": "value"}]). Must match length of ids if provided.
        documents: Optional list of new text documents.
                   Must match length of ids if provided.

    Returns:
        A confirmation message indicating the number of documents updated.

    Raises:
        ValueError: If 'ids' is empty or if none of 'embeddings', 'metadatas',
                    or 'documents' are provided, or if the length of provided
                    update lists does not match the length of 'ids'.
        Exception: If the collection does not exist or if the update operation fails.
    """
    metadatas = _parse_list_dict_param(metadatas, "metadatas")
    ids = _parse_list_param(ids, "ids")
    embeddings = _parse_list_param(embeddings, "embeddings")
    documents = _parse_list_param(documents, "documents")
    if not ids:
        raise ValueError("The 'ids' list cannot be empty.")

    if embeddings is None and metadatas is None and documents is None:
        raise ValueError(
            "At least one of 'embeddings', 'metadatas', or 'documents' "
            "must be provided for update."
        )

    # Ensure provided lists match the length of ids if they are not None
    if embeddings is not None and len(embeddings) != len(ids):
        raise ValueError("Length of 'embeddings' list must match length of 'ids' list.")
    if metadatas is not None and len(metadatas) != len(ids):
        raise ValueError("Length of 'metadatas' list must match length of 'ids' list.")
    if documents is not None and len(documents) != len(ids):
        raise ValueError("Length of 'documents' list must match length of 'ids' list.")


    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise Exception(
            f"Failed to get collection '{collection_name}': {str(e)}"
        ) from e

    # Prepare arguments for update, excluding None values at the top level
    update_args = {
        "ids": ids,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "documents": documents,
    }
    kwargs = {k: v for k, v in update_args.items() if v is not None}

    try:
        collection.update(**kwargs)
        return (
            f"Successfully processed update request for {len(ids)} documents in "
            f"collection '{collection_name}'. Note: Non-existent IDs are ignored by ChromaDB."
        )
    except Exception as e:
        raise Exception(
            f"Failed to update documents in collection '{collection_name}': {str(e)}"
        ) from e

@mcp.tool()
async def chroma_delete_documents(
    collection_name: str,
    ids: Union[str, List[str]]
) -> str:
    """Delete documents from a Chroma collection.

    Args:
        collection_name: Name of the collection to delete documents from
        ids: List of document IDs to delete, or a JSON string array (e.g., '["id1","id2"]')

    Returns:
        A confirmation message indicating the number of documents deleted.

    Raises:
        ValueError: If 'ids' is empty
        Exception: If the collection does not exist or if the delete operation fails.
    """
    if not ids:
        raise ValueError("The 'ids' list cannot be empty.")

    ids = _parse_list_param(ids, "ids")
    client = get_chroma_client()
    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        raise Exception(
            f"Failed to get collection '{collection_name}': {str(e)}"
        ) from e

    try:
        collection.delete(ids=ids)
        return (
            f"Successfully deleted {len(ids)} documents from "
            f"collection '{collection_name}'. Note: Non-existent IDs are ignored by ChromaDB."
        )
    except Exception as e:
        raise Exception(
            f"Failed to delete documents from collection '{collection_name}': {str(e)}"
        ) from e

def validate_thought_data(input_data: Dict) -> Dict:
    """Validate thought data structure."""
    if not input_data.get("sessionId"):
        raise ValueError("Invalid sessionId: must be provided")
    if not input_data.get("thought") or not isinstance(input_data.get("thought"), str):
        raise ValueError("Invalid thought: must be a string")
    if not input_data.get("thoughtNumber") or not isinstance(input_data.get("thoughtNumber"), int):
            raise ValueError("Invalid thoughtNumber: must be a number")
    if not input_data.get("totalThoughts") or not isinstance(input_data.get("totalThoughts"), int):
        raise ValueError("Invalid totalThoughts: must be a number")
    if not isinstance(input_data.get("nextThoughtNeeded"), bool):
        raise ValueError("Invalid nextThoughtNeeded: must be a boolean")
        
    return {
        "sessionId": input_data.get("sessionId"),
        "thought": input_data.get("thought"),
        "thoughtNumber": input_data.get("thoughtNumber"),
        "totalThoughts": input_data.get("totalThoughts"),
        "nextThoughtNeeded": input_data.get("nextThoughtNeeded"),
        "isRevision": input_data.get("isRevision"),
        "revisesThought": input_data.get("revisesThought"),
        "branchFromThought": input_data.get("branchFromThought"),
        "branchId": input_data.get("branchId"),
        "needsMoreThoughts": input_data.get("needsMoreThoughts"),
    }

def main():
    """Entry point for the Chroma MCP server."""
    parser = create_parser()
    args = parser.parse_args()
    
    if args.dotenv_path:
        load_dotenv(dotenv_path=args.dotenv_path)
        # re-parse args to read the updated environment variables
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
    try:
        get_chroma_client(args)
        print("Successfully initialized Chroma client")
    except Exception as e:
        print(f"Failed to initialize Chroma client: {str(e)}")
        raise
    
    # Initialize and run the server
    print("Starting MCP server")
    mcp.run(transport='stdio')
    
if __name__ == "__main__":
    main()
