import pytest
from chroma_mcp.server import get_chroma_client, create_parser, mcp
import chromadb
import sys
import os
from unittest.mock import patch, MagicMock
import argparse

# Add pytest-asyncio marker
pytest_plugins = ['pytest_asyncio']

@pytest.fixture(autouse=True)
def setup_test_args():
    # Modify sys.argv to provide the required arguments for all tests
    original_argv = sys.argv.copy()
    sys.argv = ['chroma-mcp', '--client-type', 'ephemeral']
    yield
    sys.argv = original_argv

@pytest.fixture
def mock_env_vars():
    """Fixture to mock environment variables and clean them up after tests."""
    original_environ = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(original_environ)

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

# New tests for argument parsing

def test_create_parser_defaults():
    """Test that the parser creates default values correctly."""
    parser = create_parser()
    args = parser.parse_args(['--client-type', 'ephemeral'])
    
    # Check default values
    assert args.client_type == 'ephemeral'
    assert args.ssl is True  # Default should be True
    assert args.dotenv_path == '.chroma_env'

def test_create_parser_all_args():
    """Test that the parser handles all arguments correctly."""
    parser = create_parser()
    args = parser.parse_args([
        '--client-type', 'http',
        '--host', 'test-host',
        '--port', '8080',
        '--ssl', 'false',
        '--dotenv-path', 'custom.env'
    ])
    
    # Check parsed values
    assert args.client_type == 'http'
    assert args.host == 'test-host'
    assert args.port == '8080'
    assert args.ssl is False
    assert args.dotenv_path == 'custom.env'

def test_create_parser_boolean_args():
    """Test that boolean arguments are parsed correctly with different formats."""
    parser = create_parser()
    
    # Test various true values
    for true_val in ['true', 'yes', '1', 't', 'y', 'True', 'YES']:
        args = parser.parse_args(['--client-type', 'ephemeral', '--ssl', true_val])
        assert args.ssl is True, f"Failed for value: {true_val}"
    
    # Test various false values
    for false_val in ['false', 'no', '0', 'f', 'n', 'False', 'NO']:
        args = parser.parse_args(['--client-type', 'ephemeral', '--ssl', false_val])
        assert args.ssl is False, f"Failed for value: {false_val}"

@patch.dict(os.environ, {
    'CHROMA_CLIENT_TYPE': 'http',
    'CHROMA_HOST': 'env-host',
    'CHROMA_PORT': '9090',
    'CHROMA_SSL': 'false'
})
def test_env_vars_override_defaults():
    """Test that environment variables override default values."""
    parser = create_parser()
    args = parser.parse_args([])  # No command line args
    
    # Environment variables should be used
    assert args.client_type == 'http'
    assert args.host == 'env-host'
    assert args.port == '9090'
    assert args.ssl is False

def test_cmd_args_override_env_vars(mock_env_vars):
    """Test that command line arguments override environment variables."""
    # Set environment variables
    os.environ['CHROMA_CLIENT_TYPE'] = 'http'
    os.environ['CHROMA_HOST'] = 'env-host'
    os.environ['CHROMA_SSL'] = 'false'
    
    parser = create_parser()
    # Override with command line args
    args = parser.parse_args([
        '--client-type', 'persistent',
        '--data-dir', '/test/dir',
        '--ssl', 'true'
    ])
    
    # Command line args should take precedence
    assert args.client_type == 'persistent'
    assert args.data_dir == '/test/dir'
    assert args.ssl is True
    # But other env vars should still be used
    assert args.host == 'env-host'

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.HttpClient')
def test_http_client_creation(mock_http_client, mock_env_vars):
    """Test HTTP client creation with various arguments."""
    mock_instance = MagicMock()
    mock_http_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'http',
                '--host', 'test-host',
                '--port', '8080',
                '--ssl', 'false']
    
    client = get_chroma_client()
    
    # Check that HttpClient was called with correct args
    mock_http_client.assert_called_once()
    call_kwargs = mock_http_client.call_args.kwargs
    assert call_kwargs['host'] == 'test-host'
    assert call_kwargs['port'] == '8080'
    assert call_kwargs['ssl'] is False

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.HttpClient')
def test_cloud_client_creation(mock_http_client, mock_env_vars):
    """Test cloud client creation with various arguments."""
    mock_instance = MagicMock()
    mock_http_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'cloud',
                '--tenant', 'test-tenant',
                '--database', 'test-db',
                '--api-key', 'test-api-key']
    
    client = get_chroma_client()
    
    # Check that HttpClient was called with correct args
    mock_http_client.assert_called_once()
    call_kwargs = mock_http_client.call_args.kwargs
    assert call_kwargs['host'] == 'api.trychroma.com'
    assert call_kwargs['ssl'] is True  # Always true for cloud
    assert call_kwargs['tenant'] == 'test-tenant'
    assert call_kwargs['database'] == 'test-db'
    assert call_kwargs['headers'] == {'x-chroma-token': 'test-api-key'}

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.PersistentClient')
def test_persistent_client_creation(mock_persistent_client, mock_env_vars):
    """Test persistent client creation."""
    mock_instance = MagicMock()
    mock_persistent_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', 
                '--client-type', 'persistent',
                '--data-dir', '/test/data/dir']
    
    client = get_chroma_client()
    
    # Check that PersistentClient was called with correct args
    mock_persistent_client.assert_called_once_with(path='/test/data/dir')

@patch('chroma_mcp.server._chroma_client', None)  # Reset the global client
@patch('chromadb.EphemeralClient')
def test_ephemeral_client_creation(mock_ephemeral_client, mock_env_vars):
    """Test ephemeral client creation."""
    mock_instance = MagicMock()
    mock_ephemeral_client.return_value = mock_instance
    
    # Set up command line args
    sys.argv = ['chroma-mcp', '--client-type', 'ephemeral']
    
    client = get_chroma_client()
    
    # Check that EphemeralClient was called
    mock_ephemeral_client.assert_called_once()

def test_client_type_validation():
    """Test validation of client type argument."""
    parser = create_parser()
    
    # Valid client types
    for valid_type in ['http', 'cloud', 'persistent', 'ephemeral']:
        args = parser.parse_args(['--client-type', valid_type])
        assert args.client_type == valid_type
    
    # Invalid client type
    with pytest.raises(SystemExit):
        parser.parse_args(['--client-type', 'invalid'])

def test_required_args_for_http_client():
    """Test that required arguments are enforced for HTTP client."""
    with patch('argparse.ArgumentParser.error') as mock_error:
        from chroma_mcp.server import main
        
        # Set up command line args without required host
        sys.argv = ['chroma-mcp', '--client-type', 'http']
        
        try:
            main()
        except:
            pass
        
        # Check that error was called for missing host
        mock_error.assert_called_with(
            "Host must be provided via --host flag or CHROMA_HOST environment variable when using HTTP client"
        )

def test_required_args_for_cloud_client():
    """Test that required arguments are enforced for cloud client."""
    with patch('argparse.ArgumentParser.error') as mock_error:
        from chroma_mcp.server import main
        
        # Set up command line args without required tenant/database/api-key
        sys.argv = ['chroma-mcp', '--client-type', 'cloud']
        
        try:
            main()
        except:
            pass
        
        # Check that error was called for missing api-key (the first check in the code)
        mock_error.assert_called_with(
            "API key must be provided via --api-key flag or CHROMA_API_KEY environment variable when using cloud client"
        )