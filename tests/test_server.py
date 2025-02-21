import pytest
from chroma_mcp.server import get_chroma_client, mcp
import chromadb
import sys

# Add pytest-asyncio marker
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(autouse=True)
def setup_test_args():
    # Modify sys.argv to provide the required arguments for all tests
    original_argv = sys.argv.copy()
    sys.argv = ['chroma-mcp', '--client-type', 'ephemeral']
    yield
    sys.argv = original_argv

def test_get_chroma_client_ephemeral():
    # Test ephemeral client creation
    client = get_chroma_client()
    assert isinstance(client, chromadb.ClientAPI)

@pytest.mark.asyncio
async def test_list_collections():
    # Test list_collections tool
    result = await mcp.call_tool("list_collections", {"limit": None, "offset": None})
    assert isinstance(result, list)

@pytest.mark.asyncio
async def test_create_and_delete_collection():
    # Test collection creation and deletion
    collection_name = "test_collection"
    
    # Create collection
    create_result = await mcp.call_tool("create_collection", {"collection_name": collection_name})
    assert len(create_result) == 1  # Should return a list with one TextContent
    assert "Successfully created collection" in create_result[0].text
    
    # Delete collection
    delete_result = await mcp.call_tool("delete_collection", {"collection_name": collection_name})
    assert len(delete_result) == 1  # Should return a list with one TextContent
    assert "Successfully deleted collection" in delete_result[0].text 