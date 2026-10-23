"""Regression tests for issue #66: diagnostics must not go to stdout.

The MCP stdio transport reserves stdout exclusively for JSON-RPC frames.
Any plain-text diagnostic printed to stdout corrupts the protocol stream,
so every diagnostic print in ``chroma_mcp.server`` must target stderr.
"""
import argparse
import ast
import ssl
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import chroma_mcp.server as server_module
from chroma_mcp.server import get_chroma_client, main


def _http_args(**overrides):
    """Build the argparse.Namespace get_chroma_client expects for http."""
    args = argparse.Namespace(
        client_type="http",
        data_dir=None,
        host="test-host",
        port="8000",
        custom_auth_credentials=None,
        tenant=None,
        database=None,
        api_key=None,
        ssl=True,
        dotenv_path=".chroma_env",
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_no_print_to_stdout_in_server_source():
    """Every print() call in server.py must explicitly write to stderr."""
    source_path = Path(server_module.__file__)
    tree = ast.parse(source_path.read_text())

    offenders = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "print"
        ):
            keywords = {kw.arg: ast.unparse(kw.value) for kw in node.keywords}
            if keywords.get("file") != "sys.stderr":
                offenders.append(node.lineno)

    assert offenders == [], (
        f"print() calls without file=sys.stderr at lines: {offenders}"
    )


def test_main_startup_messages_go_to_stderr(capsys, monkeypatch):
    """main() startup diagnostics must be on stderr, keeping stdout clean."""
    monkeypatch.setattr(sys, "argv", ["chroma-mcp", "--client-type", "ephemeral"])

    with patch("chroma_mcp.server.get_chroma_client", return_value=MagicMock()), \
         patch.object(server_module.mcp, "run", return_value=None):
        main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Successfully initialized Chroma client" in captured.err
    assert "Starting MCP server" in captured.err


def test_main_failure_message_goes_to_stderr(capsys, monkeypatch):
    """A failed client initialization in main() must report on stderr."""
    monkeypatch.setattr(sys, "argv", ["chroma-mcp", "--client-type", "ephemeral"])

    with patch(
        "chroma_mcp.server.get_chroma_client",
        side_effect=Exception("boom"),
    ), patch.object(server_module.mcp, "run", return_value=None):
        with pytest.raises(Exception, match="boom"):
            main()

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Failed to initialize Chroma client: boom" in captured.err


@patch("chroma_mcp.server._chroma_client", None)  # Reset the global client
@patch("chromadb.HttpClient")
def test_http_client_error_goes_to_stderr(mock_http_client, capsys):
    """HTTP connection errors in get_chroma_client must go to stderr."""
    mock_http_client.side_effect = Exception("connection refused")

    with pytest.raises(Exception, match="connection refused"):
        get_chroma_client(_http_args())

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Error connecting to HTTP client: connection refused" in captured.err


@patch("chroma_mcp.server._chroma_client", None)  # Reset the global client
@patch("chromadb.HttpClient")
def test_cloud_client_ssl_error_goes_to_stderr(mock_http_client, capsys):
    """Cloud SSL errors in get_chroma_client must go to stderr."""
    mock_http_client.side_effect = ssl.SSLError("certificate verify failed")

    args = _http_args(
        client_type="cloud",
        tenant="test-tenant",
        database="test-db",
        api_key="test-api-key",
    )
    with pytest.raises(ssl.SSLError, match="certificate verify failed"):
        get_chroma_client(args)

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "SSL connection failed:" in captured.err
    assert "certificate verify failed" in captured.err
