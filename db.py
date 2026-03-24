"""
Database Connection Module
Provides reusable database connections for FAISS vector store and MongoDB filtering layer.
"""

from dotenv import load_dotenv
import os
from typing import Optional, Dict, Any, List
from pymongo import MongoClient
from pymongo.errors import ServerSelectionTimeoutError, ConnectionFailure
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configuration
MONGODB_URI: str = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
MONGODB_DB: str = os.getenv("MONGODB_DB_NAME", "rag_filtering")
MONGODB_COLLECTION: str = os.getenv("MONGODB_COLLECTION", "filters")

# Global MongoDB connection instance (Singleton pattern)
_mongo_client: Optional[MongoClient] = None
_mongo_db: Optional[Any] = None


def get_mongo_client() -> MongoClient:
    """
    Get or create MongoDB client instance.
    Uses singleton pattern to maintain a single connection.

    Returns:
        MongoClient: MongoDB client instance

    Raises:
        ConnectionFailure: If unable to connect to MongoDB
    """
    global _mongo_client

    if _mongo_client is None:
        try:
            _mongo_client = MongoClient(
                MONGODB_URI,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=10000
            )
            # Verify connection
            _mongo_client.admin.command('ping')
            logger.info("✓ MongoDB connected successfully")
        except (ServerSelectionTimeoutError, ConnectionFailure) as e:
            logger.error(f"✗ Failed to connect to MongoDB: {e}")
            raise ConnectionFailure(f"Could not connect to MongoDB at {MONGODB_URI}") from e

    return _mongo_client


def get_mongo_db(db_name: str = MONGODB_DB) -> Any:
    """
    Get MongoDB database instance.

    Args:
        db_name (str): Database name

    Returns:
        Database: MongoDB database object
    """
    global _mongo_db

    if _mongo_db is None:
        client = get_mongo_client()
        _mongo_db = client[db_name]

    print(f"Using MongoDB database: {db_name}")
    return _mongo_db


def get_mongo_collection(collection_name: str = MONGODB_COLLECTION) -> Any:
    """
    Get MongoDB collection instance.

    Args:
        collection_name (str): Collection name

    Returns:
        Collection: MongoDB collection object
    """
    db = get_mongo_db()
    return db[collection_name]


if __name__ == "__main__":
    # Test MongoDB connection
    try:
        client = get_mongo_client()
        db = get_mongo_db()
        collection = get_mongo_collection()
        logger.info(f"MongoDB connection test successful: {client}, DB: {db.name}, Collection: {collection.name}")
    except Exception as e:
        logger.error(f"MongoDB connection test failed: {e}")


def create_filter_document(
    query_id: str,
    query_text: str,
    filters: Dict[str, Any],
    metadata: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Create a filter document for MongoDB storage.

    Args:
        query_id (str): Unique query identifier
        query_text (str): The query text
        filters (Dict): Filter criteria
        metadata (Dict): Additional metadata

    Returns:
        Dict: Document ready for MongoDB insertion
    """
    from datetime import datetime

    document = {
        "query_id": query_id,
        "query_text": query_text,
        "filters": filters,
        "metadata": metadata or {},
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow()
    }
    return document


def insert_filter(filter_doc: Dict[str, Any], collection_name: str = MONGODB_COLLECTION) -> str:
    """
    Insert a filter document into MongoDB.

    Args:
        filter_doc (Dict): Filter document to insert
        collection_name (str): Target collection name

    Returns:
        str: Inserted document ID
    """
    try:
        collection = get_mongo_collection(collection_name)
        result = collection.insert_one(filter_doc)
        logger.info(f"✓ Filter inserted with ID: {result.inserted_id}")
        return str(result.inserted_id)
    except Exception as e:
        logger.error(f"✗ Error inserting filter: {e}")
        raise


def find_filter(query: Dict[str, Any], collection_name: str = MONGODB_COLLECTION) -> Optional[Dict]:
    """
    Find a single filter document.

    Args:
        query (Dict): MongoDB query filter
        collection_name (str): Target collection name

    Returns:
        Dict or None: Found document or None
    """
    try:
        collection = get_mongo_collection(collection_name)
        result = collection.find_one(query)
        return result
    except Exception as e:
        logger.error(f"✗ Error finding filter: {e}")
        return None


def find_filters(query: Dict[str, Any], collection_name: str = MONGODB_COLLECTION) -> List[Dict]:
    """
    Find multiple filter documents.

    Args:
        query (Dict): MongoDB query filter
        collection_name (str): Target collection name

    Returns:
        List[Dict]: List of found documents
    """
    try:
        collection = get_mongo_collection(collection_name)
        results = list(collection.find(query))
        return results
    except Exception as e:
        logger.error(f"✗ Error finding filters: {e}")
        return []


def update_filter(
    query: Dict[str, Any],
    update_data: Dict[str, Any],
    collection_name: str = MONGODB_COLLECTION
) -> int:
    """
    Update a filter document.

    Args:
        query (Dict): MongoDB query filter
        update_data (Dict): Data to update
        collection_name (str): Target collection name

    Returns:
        int: Number of documents modified
    """
    try:
        from datetime import datetime
        collection = get_mongo_collection(collection_name)
        update_data["updated_at"] = datetime.utcnow()
        result = collection.update_one(query, {"$set": update_data})
        logger.info(f"✓ Updated {result.modified_count} filter(s)")
        return result.modified_count
    except Exception as e:
        logger.error(f"✗ Error updating filter: {e}")
        raise


def delete_filter(query: Dict[str, Any], collection_name: str = MONGODB_COLLECTION) -> int:
    """
    Delete a filter document.

    Args:
        query (Dict): MongoDB query filter
        collection_name (str): Target collection name

    Returns:
        int: Number of documents deleted
    """
    try:
        collection = get_mongo_collection(collection_name)
        result = collection.delete_one(query)
        logger.info(f"✓ Deleted {result.deleted_count} filter(s)")
        return result.deleted_count
    except Exception as e:
        logger.error(f"✗ Error deleting filter: {e}")
        raise


def close_mongo_connection():
    """Close MongoDB connection."""
    global _mongo_client, _mongo_db

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
        _mongo_db = None
        logger.info("✓ MongoDB connection closed")


class MongoDBConnection:
    """Context manager for MongoDB connections."""

    def __init__(self, collection_name: str = MONGODB_COLLECTION):
        self.collection_name = collection_name
        self.collection = None

    def __enter__(self):
        self.collection = get_mongo_collection(self.collection_name)
        return self.collection

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup if needed"""
        pass


# Example usage functions
def apply_filters_to_results(
    search_results: List[Dict],
    filter_query: Dict[str, Any],
    collection_name: str = MONGODB_COLLECTION
) -> List[Dict]:
    """
    Apply stored filters to search results for filtering layer.

    Args:
        search_results (List[Dict]): Original search results
        filter_query (Dict): Query to find applicable filters
        collection_name (str): Target collection name

    Returns:
        List[Dict]: Filtered results
    """
    filters = find_filters(filter_query, collection_name)

    if not filters:
        return search_results

    filtered_results = []
    for result in search_results:
        should_include = True
        for filter_doc in filters:
            # Apply filter logic here based on your requirements
            filter_conditions = filter_doc.get("filters", {})
            for key, value in filter_conditions.items():
                if key in result and result[key] != value:
                    should_include = False
                    break

        if should_include:
            filtered_results.append(result)

    return filtered_results
