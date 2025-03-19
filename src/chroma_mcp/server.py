from typing import Dict, List, Optional
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
        List of collection names
    """
    client = get_chroma_client()
    return client.list_collections(limit=limit, offset=offset)


@mcp.tool()
async def chroma_create_collection(
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
    collection = client.get_collection(collection_name)
    results = collection.peek(limit=limit)
    return results

@mcp.tool()
async def chroma_get_collection_info(collection_name: str) -> Dict:
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
async def chroma_get_collection_count(collection_name: str) -> int:
    """Get the number of documents in a Chroma collection.
    
    Args:
        collection_name: Name of the collection to count
    """
    client = get_chroma_client()
    collection = client.get_collection(collection_name)
    return collection.count()

@mcp.tool()
async def chroma_modify_collection(
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
async def chroma_delete_collection(collection_name: str) -> str:
    """Delete a Chroma collection.
    
    Args:
        collection_name: Name of the collection to delete
    """
    client = get_chroma_client()
    client.delete_collection(collection_name)
    return f"Successfully deleted collection {collection_name}"

##### Document Tools #####
@mcp.tool()
async def chroma_add_documents(
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
async def chroma_query_documents(
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
async def chroma_get_documents(
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
    
def process_thought(input_data: Dict) -> Dict:
    """Process a new thought."""
    try:
        # Validate input data
        validated_input = validate_thought_data(input_data)
            
        # Adjust total thoughts if needed
        if validated_input["thoughtNumber"] > validated_input["totalThoughts"]:
            validated_input["totalThoughts"] = validated_input["thoughtNumber"]
            
        # Return response
        return {
            "sessionId": validated_input["sessionId"],
            "thoughtNumber": validated_input["thoughtNumber"],
            "totalThoughts": validated_input["totalThoughts"],
            "nextThoughtNeeded": validated_input["nextThoughtNeeded"],
            }
            
    except Exception as e:
        return {
            "error": str(e),
            "status": "failed"
        }

# Initialize persistence
def init_thinking_collections(client):
    """Initialize collections needed for sequential thinking."""
    # Main collection for thought history
    return client.get_or_create_collection(
        name="sequential_thinking",
    ), client.get_or_create_collection(
        name="sequential_thinking_branches",
    ), client.get_or_create_collection(
        name="sequential_thinking_summary",
    )
    
def find_similar_sessions(thoughts_collection, summary_collection, thought, similarity_threshold=0.75, max_results=3):
    """Find similar sessions based on thought content."""
    
    # Query similar thoughts
    similar_thoughts = thoughts_collection.query(
        query_texts=[thought], 
        n_results=max_results,
        include=["metadatas", "documents", "distances"]
    )
    
    context_results = []
    
    # Process results
    for i in range(len(similar_thoughts["ids"][0])):
        distance = similar_thoughts["distances"][0][i]
        
        # Only include results that meet threshold
        if distance < similarity_threshold:
            metadata = similar_thoughts["metadatas"][0][i]
            session_id = metadata["sid"]
            
            # Get session summary
            summary_results = summary_collection.get(
                ids=[session_id],
                include=["documents", "metadatas"]
            )
            
            if summary_results["documents"]:
                summary = summary_results["documents"][0]
                summary_metadata = summary_results["metadatas"][0] if summary_results["metadatas"] else {}
                
                # Calculate relevance score (1.0 is exact match, 0.0 is completely dissimilar)
                relevance_score = 1.0 - distance
                
                context_results.append({
                    "sessionId": session_id,
                    "summary": summary,
                    "relevanceScore": round(relevance_score, 2),
                    "similarThought": similar_thoughts["documents"][0][i],
                    "thoughtNumber": metadata["tn"],
                    "keyThoughts": json.loads(summary_metadata.get("keyThoughts", "[]")) if summary_metadata else [],
                })
    
    return context_results


@mcp.tool()
async def chroma_sequential_thinking(
    thought: str,
    thoughtNumber: int,
    totalThoughts: int,
    nextThoughtNeeded: bool,
    sessionId: Optional[str] = None,
    isRevision: Optional[bool] = None,
    revisesThought: Optional[int] = None,
    branchFromThought: Optional[int] = None,
    branchId: Optional[str] = None,
    needsMoreThoughts: Optional[bool] = None,
    sessionSummary: Optional[str] = None,
    keyThoughts: Optional[List[int]] = None,
    persist: bool = True
) -> Dict:
    """A detailed tool for dynamic and reflective problem-solving through thoughts.
    
    This tool helps analyze problems through a flexible thinking process that can adapt and evolve.
    Each thought can build on, question, or revise previous insights as understanding deepens.
    
    Args:
        thought: Do not store code in the thought. It is your current thinking step, which can include:
            * Regular analytical steps
            * Revisions of previous thoughts
            * Questions about previous decisions
            * Realizations about needing more analysis
            * Changes in approach
            * Hypothesis generation
            * Hypothesis verification
        Do not make thoughts superfluous. Do not store code in the thought. 
        thoughtNumber: Current thought number
            * The current number in sequence (can go beyond initial total if needed)
        totalThoughts: Estimated total thoughts needed
            * Current estimate of thoughts needed (can be adjusted up/down)
        sessionId: Identifier for the thinking session. Provide if this is not the first thought in the session.
            * A unique identifier for the current thinking session
        nextThoughtNeeded: Whether another thought step is needed
            * True if you need more thinking, even if at what seemed like the end
        isRevision: Whether this revises previous thinking
            * A boolean indicating if this thought revises previous thinking
        revisesThought: Which thought is being reconsidered
            * If isRevision is true, which thought number is being reconsidered
        branchFromThought: Branching point thought number
            * If branching, which thought number is the branching point
        branchId: Branch identifier
            * Identifier for the current branch (if any)
        needsMoreThoughts: If more thoughts are needed
            * If reaching end but realizing more thoughts needed
        sessionSummary: A summary of the current session. Provide when nextThoughtNeeded is false.
        keyThoughts: A list of key thought numbers from the current session. Provide when nextThoughtNeeded is false.
        persist: Whether to persist thoughts in the Chroma database
        
        You should:
        1. Start with an initial estimate of needed thoughts, but be ready to adjust
        2. Feel free to question or revise previous thoughts
        3. Don't hesitate to add more thoughts if needed, even at the "end"
        4. Express uncertainty when present
        5. Mark thoughts that revise previous thinking or branch into new paths
        6. Ignore information that is irrelevant to the current step
        7. Generate a solution hypothesis when appropriate
        8. Verify the hypothesis based on the Chain of Thought steps
        9. Repeat the process until satisfied with the solution
        10. Provide a single, ideally correct answer as the final output
        11. Only set next_thought_needed to false when truly done and a satisfactory answer is reached`,
    
    Returns:
        Dictionary with thought metadata
    """
    current_time = time.time()
    
    # Generate session ID if not provided
    if not sessionId:
        sessionId = str(uuid.uuid4())[:8]
    
    # Generate branch ID if branching but no ID provided
    if branchFromThought and not branchId:
        branchId = str(uuid.uuid4())[:8]
    
    # Structure input data
    input_data = {
        "sessionId": sessionId,
        "thought": thought,
        "thoughtNumber": thoughtNumber,
        "totalThoughts": totalThoughts,
        "nextThoughtNeeded": nextThoughtNeeded,
        "isRevision": isRevision,
        "revisesThought": revisesThought,
        "branchFromThought": branchFromThought,
        "branchId": branchId,
        "needsMoreThoughts": needsMoreThoughts,
    }
    
    # Process the thought
    processed_thought = process_thought(input_data)
    
    if persist:
        client = get_chroma_client()
        thoughts_collection, branches_collection, summary_collection = init_thinking_collections(client)
        
        # Find similar sessions on first thought
        if thoughtNumber == 1:
            similar_sessions = find_similar_sessions(thoughts_collection, summary_collection, thought)
        else:
            similar_sessions = []
        processed_thought["context"] = similar_sessions
        
        # Store the thought
        thought_id = f"{sessionId}_{thoughtNumber}"
        if branchId:
            thought_id = f"{thought_id}_{branchId}"
        
        # Add metadata
        metadata = {
            "sid": sessionId,
            "tn": thoughtNumber,
            "tt": totalThoughts,
            "isR": bool(isRevision),
            "rt": revisesThought if revisesThought else -1,
            "bf": branchFromThought if branchFromThought else -1,
            "bid": branchId if branchId else "",
            "ntn": nextThoughtNeeded,
            "ts": current_time,
        }
        
        # Store in Chroma
        thoughts_collection.add(
            documents=[thought],
            metadatas=[metadata],
            ids=[thought_id]
        )
        
        # Add branch relationship if applicable
        if branchFromThought and branchId:
            branch_metadata = {
                "sid": sessionId,
                "pt": branchFromThought,
                "bid": branchId,
                "ts": current_time,
            }
            
            branches_collection.add(
                documents=[f"Branch from thought {branchFromThought} in session {sessionId}"],
                metadatas=[branch_metadata],
                ids=[f"branch_{branchId}"]
            )
        
        # Add persistence info to result
        processed_thought["persistedId"] = thought_id
        
        if sessionSummary:
            existing_summary = None
            try:
                summary_result = summary_collection.get(ids=[sessionId], include=["documents", "metadatas"])
                if summary_result["documents"]:
                    existing_summary = summary_result["documents"][0]
                    existing_metadata = summary_result["metadatas"][0]
            except:
                pass
            summary_metadata = {
                "sid": sessionId,
                "created_ts": current_time,
                "updated_ts": current_time,
                "version": 1,
                "keyThoughts": json.dumps(keyThoughts) if keyThoughts else None,
            }
            
            if existing_summary:
                summary_metadata["version"] = existing_metadata.get("version", 0) + 1
                summary_metadata["created_ts"] = existing_metadata.get("created_ts", current_time)
            
                summary_collection.update(
                    ids=[sessionId],
                    documents=[sessionSummary],
                    metadatas=[summary_metadata]
                )
            else:
                summary_collection.add(
                    documents=[sessionSummary],
                    metadatas=[summary_metadata],
                    ids=[sessionId]
                )
            processed_thought["summary"] = {
                "content": sessionSummary,
                "version": summary_metadata["version"],
            }
    
    # Return processed result as JSON
    return processed_thought

@mcp.tool()
async def chroma_get_similar_sessions(
    text: str,
) -> Dict:
    """Retrieve the thought history for a specific session.
    
    Args:
        text: The text to search for
    
    Returns:
        Dictionary with the thought history
    """
    client = get_chroma_client()
    thoughts_collection = client.get_collection("sequential_thinking_summary")
    
    # Query for similar sessions
    results = thoughts_collection.query(
        query_texts=[text],
        n_results=5,
        include=["documents", "metadatas"]
    )
    
    return results


@mcp.tool()
async def chroma_get_thought_history(
    sessionId: str,
) -> Dict:
    """Retrieve the thought history for a specific session.
    
    Args:
        sessionId: The session identifier
    
    Returns:
        Dictionary with the thought history
    """
    client = get_chroma_client()
    thoughts_collection = client.get_collection("sequential_thinking")
    
    # Query for thoughts in this session
    results = thoughts_collection.get(
        where={"sid": sessionId},
        include=["documents", "metadatas"]
    )
    
    # Sort by thought number
    thoughts = []
    for i, doc in enumerate(results["documents"]):
        metadata = results["metadatas"][i]
        thoughts.append({
            "thought": doc,
            "metadata": metadata
        })
    
    # Sort thoughts by number
    thoughts.sort(key=lambda x: x["metadata"]["tn"])
    
    return {
        "sessionId": sessionId,
        "thoughts": thoughts,
        "totalThoughts": len(thoughts)
    }

@mcp.tool()
async def chroma_get_thought_branches(sessionId: str) -> Dict:
    """Get all branches for a specific thinking session.
    
    Args:
        sessionId: The session identifier
    
    Returns:
        Dictionary with branch information
    """
    client = get_chroma_client()
    branches_collection = client.get_collection("sequential_thinking_branches")
    
    # Query for branches in this session
    results = branches_collection.get(
        where={"sid": sessionId},
        include=["metadatas"]
    )
    
    branches = []
    for metadata in results["metadatas"]:
        branches.append({
            "branchId": metadata["bid"],
            "parentThought": metadata["pt"]
        })
    
    return {
        "sessionId": sessionId,
        "branches": branches,
        "totalBranches": len(branches)
    }

@mcp.tool()
async def chroma_continue_thought_chain(
    sessionId: str,
    branchId: Optional[str] = None
) -> Dict:
    """Get the latest state of a thought chain to continue it.
    
    Args:
        sessionId: The session identifier
        branchId: Optional branch identifier to continue a specific branch
    
    Returns:
        Dictionary with the latest state of the thought chain
    """
    client = get_chroma_client()
    thoughts_collection = client.get_collection("sequential_thinking")
    
    # Build query
    where_clause = {"sid": sessionId}
    if branchId:
        where_clause["branchId"] = branchId
    
    # Get all thoughts in the chain
    results = thoughts_collection.get(
        where=where_clause,
        include=["documents", "metadatas"]
    )
    
    if not results["documents"]:
        return {
            "error": f"No thoughts found for session {sessionId}" + (f" and branch {branchId}" if branchId else ""),
            "status": "failed"
        }
    
    # Find the highest thought number
    max_thought_num = 0
    latest_thought = None
    for i, metadata in enumerate(results["metadatas"]):
        if metadata["tn"] > max_thought_num:
            max_thought_num = metadata["tn"]
            latest_thought = {
                "thought": results["documents"][i],
                "metadata": metadata
            }
    
    # Return state to continue from
    return {
        "sessionId": sessionId,
        "latestThought": latest_thought["thought"],
        "thoughtNumber": latest_thought["metadata"]["tn"],
        "totalThoughts": latest_thought["metadata"]["tt"],
        "nextThoughtNumber": latest_thought["metadata"]["tn"] + 1,
        "branchId": branchId if branchId else latest_thought["metadata"].get("bid", ""),
        "nextThoughtNeeded": latest_thought["metadata"]["ntn"]
    }

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
