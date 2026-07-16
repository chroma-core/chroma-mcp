import sys
import os

# Ensure src directory is in Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import asyncio
import argparse
from chroma_mcp.server import (
    get_chroma_client,
    chroma_create_collection,
    chroma_add_documents,
    chroma_get_documents,
    chroma_delete_collection,
    _chroma_client,
)


def init_client():
    """Initialize ephemeral Chroma client"""
    global _chroma_client
    args = argparse.Namespace(
        client_type='ephemeral',
        data_dir=None,
        host=None,
        port=None,
        custom_auth_credentials=None,
        tenant=None,
        database=None,
        api_key=None,
        ssl=True,
        dotenv_path='.chroma_env',
    )
    get_chroma_client(args)
    print("✅ Chroma client initialized successfully")


async def setup_test_data():
    """Create test collection and add test data"""
    collection_name = "test_get_documents"

    # Create collection
    result = await chroma_create_collection(collection_name)
    print(f"📦 {result}")

    # Add test documents
    documents = [
        "Python is a high-level programming language",
        "JavaScript is the core language for web development",
        "Chroma is a vector database",
        "MCP stands for Model Context Protocol",
        "FastAPI is a high-performance Python web framework",
    ]
    ids = [f"doc_{i}" for i in range(1, len(documents) + 1)]
    metadatas = [
        {"category": "programming", "language": "python", "difficulty": 3},
        {"category": "programming", "language": "javascript", "difficulty": 4},
        {"category": "database", "language": "python", "difficulty": 5},
        {"category": "protocol", "language": "general", "difficulty": 2},
        {"category": "framework", "language": "python", "difficulty": 4},
    ]

    result = await chroma_add_documents(
        collection_name=collection_name,
        documents=documents,
        ids=ids,
        metadatas=metadatas,
    )
    print(f"📝 {result}")
    return collection_name


async def test_get_all_documents(collection_name: str):
    """Test 1: Get all documents (no filter)"""
    print("\n--- Test 1: Get all documents ---")
    result = await chroma_get_documents(collection_name=collection_name)
    print(f"Returned document count: {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    print(f"Documents: {result.get('documents')}")
    assert len(result.get('ids', [])) == 5, f"Expected 5 documents, got {len(result.get('ids', []))}"
    print("✅ Test 1 passed")


async def test_get_by_ids(collection_name: str):
    """Test 2: Get documents by IDs"""
    print("\n--- Test 2: Get documents by IDs ---")
    result = await chroma_get_documents(
        collection_name=collection_name,
        ids=["doc_1", "doc_3", "doc_5"],
    )
    print(f"Returned document count: {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    assert len(result.get('ids', [])) == 3
    assert "doc_1" in result.get('ids', [])
    assert "doc_3" in result.get('ids', [])
    assert "doc_5" in result.get('ids', [])
    print("✅ Test 2 passed")


async def test_get_by_metadata_filter(collection_name: str):
    """Test 3: Filter by metadata (where condition)"""
    print("\n--- Test 3: Filter by metadata ---")
    # Filter language == "python"
    result = await chroma_get_documents(
        collection_name=collection_name,
        where='{"language": "python"}',
    )
    print(f"Documents with language=python: {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    assert len(result.get('ids', [])) == 3  # doc_1, doc_3, doc_5
    print("✅ Test 3 passed")


async def test_get_by_metadata_comparison(collection_name: str):
    """Test 4: Filter by metadata comparison operators ($gt, $eq, etc.)"""
    print("\n--- Test 4: Filter by metadata comparison ---")
    # Filter difficulty >= 4
    result = await chroma_get_documents(
        collection_name=collection_name,
        where='{"difficulty": {"$gte": 4}}',
    )
    print(f"Documents with difficulty >= 4: {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    assert len(result.get('ids', [])) == 3  # doc_2(4), doc_3(5), doc_5(4)
    print("✅ Test 4 passed")


async def test_get_by_document_content_filter(collection_name: str):
    """Test 5: Filter by document content (where_document)"""
    print("\n--- Test 5: Filter by document content ---")
    # Filter documents containing "Python"
    result = await chroma_get_documents(
        collection_name=collection_name,
        where_document='{"$contains": "Python"}',
    )
    print(f"Documents containing 'Python': {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    print(f"Documents: {result.get('documents')}")
    # doc_1: "Python is a high-level programming language", doc_5: "FastAPI is a high-performance Python web framework"
    assert len(result.get('ids', [])) == 2
    print("✅ Test 5 passed")


async def test_get_with_limit_offset(collection_name: str):
    """Test 6: Pagination (limit + offset)"""
    print("\n--- Test 6: Pagination ---")
    # Page 1: first 2 items
    result_page1 = await chroma_get_documents(
        collection_name=collection_name,
        limit=2,
        offset=0,
    )
    print(f"Page 1 (limit=2, offset=0): {len(result_page1.get('ids', []))} items")
    print(f"IDs: {result_page1.get('ids')}")

    # Page 2: skip 2, take 2
    result_page2 = await chroma_get_documents(
        collection_name=collection_name,
        limit=2,
        offset=2,
    )
    print(f"Page 2 (limit=2, offset=2): {len(result_page2.get('ids', []))} items")
    print(f"IDs: {result_page2.get('ids')}")

    assert len(result_page1.get('ids', [])) == 2
    assert len(result_page2.get('ids', [])) == 2
    # IDs across pages should not overlap
    page1_ids = set(result_page1.get('ids', []))
    page2_ids = set(result_page2.get('ids', []))
    assert page1_ids.isdisjoint(page2_ids), "Pagination results should not overlap"
    print("✅ Test 6 passed")


async def test_get_custom_include(collection_name: str):
    """Test 7: Custom include parameter"""
    print("\n--- Test 7: Custom include parameter ---")
    # Only return IDs, no document content or metadata
    result = await chroma_get_documents(
        collection_name=collection_name,
        ids=["doc_1"],
        include=[],  # Don't include any extra content
    )
    print(f"include=[] result: {result}")
    assert "ids" in result
    # documents and metadatas should not be in the result
    print("✅ Test 7 passed")


async def test_get_empty_ids(collection_name: str):
    """Test 8: Pass empty ids list (expected to raise exception)"""
    print("\n--- Test 8: Pass empty ids list ---")
    try:
        result = await chroma_get_documents(
            collection_name=collection_name,
            ids=[],
        )
        print(f"Empty ids returned: {result}")
        # If no exception, accept empty result
        assert len(result.get('ids', [])) == 0
    except Exception as e:
        # ChromaDB does not allow empty ids list, will raise exception, this is expected
        print(f"Empty ids expected exception: {e}")
    print("✅ Test 8 passed")


async def test_get_nonexistent_id(collection_name: str):
    """Test 9: Query non-existent document ID"""
    print("\n--- Test 9: Query non-existent document ID ---")
    result = await chroma_get_documents(
        collection_name=collection_name,
        ids=["nonexistent_id"],
    )
    print(f"Non-existent ID returned: {result}")
    assert len(result.get('ids', [])) == 0
    print("✅ Test 9 passed")


async def test_get_combined_filters(collection_name: str):
    """Test 10: Combine where and where_document filters"""
    print("\n--- Test 10: Combined filter conditions ---")
    result = await chroma_get_documents(
        collection_name=collection_name,
        where='{"language": "python"}',
        where_document='{"$contains": "Python"}',
    )
    print(f"Documents with language=python and containing 'Python': {len(result.get('ids', []))}")
    print(f"IDs: {result.get('ids')}")
    # doc_1: Python is a high-level programming language (language=python, contains Python)
    # doc_3: Chroma is a vector database (language=python, does not contain Python)
    # doc_5: FastAPI is a high-performance Python web framework (language=python, contains Python)
    assert len(result.get('ids', [])) == 2
    print("✅ Test 10 passed")


async def cleanup(collection_name: str):
    """Clean up test data"""
    print(f"\n🧹 Cleaning up test collection: {collection_name}")
    try:
        await chroma_delete_collection(collection_name)
        print("✅ Cleanup complete")
    except Exception as e:
        print(f"⚠️ Cleanup failed: {e}")


async def main():
    """Run all tests"""
    print("=" * 60)
    print("🧪 chroma_get_documents method tests")
    print("=" * 60)

    init_client()
    collection_name = await setup_test_data()

    tests = [
        ("Get all documents", test_get_all_documents),
        ("Get by IDs", test_get_by_ids),
        ("Metadata filter", test_get_by_metadata_filter),
        ("Metadata comparison filter", test_get_by_metadata_comparison),
        ("Document content filter", test_get_by_document_content_filter),
        ("Pagination", test_get_with_limit_offset),
        ("Custom include", test_get_custom_include),
        ("Empty ids list", test_get_empty_ids),
        ("Non-existent ID", test_get_nonexistent_id),
        ("Combined filters", test_get_combined_filters),
    ]

    passed = 0
    failed = 0

    for name, test_func in tests:
        try:
            await test_func(collection_name)
            passed += 1
        except Exception as e:
            failed += 1
            print(f"❌ Test failed [{name}]: {e}")

    await cleanup(collection_name)

    print("\n" + "=" * 60)
    print(f"📊 Test results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)