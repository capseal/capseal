# Action: insert_after
# Insert this method in the SQLiteCache class

async def health_check(self) -> dict[str, Any]:
    """
    Perform a health check on the SQLite cache.
    
    Returns:
        Dict containing cache health status, hit count, and miss count
    """
    try:
        # Test database connectivity by performing a simple query
        cursor = self._conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cache_entries")
        entry_count = cursor.fetchone()[0]
        
        # Check for expired entries
        now = datetime.utcnow().timestamp()
        cursor.execute("SELECT COUNT(*) FROM cache_entries WHERE expires_at > ?", (now,))
        valid_entries = cursor.fetchone()[0]
        
        # Calculate cache size (approximate)
        cursor.execute("SELECT page_count * page_size FROM pragma_page_count(), pragma_page_size()")
        size_bytes = cursor.fetchone()[0]
        
        return {
            "status": "healthy",
            "hit_count": getattr(self, '_hits', 0),
            "miss_count": getattr(self, '_misses', 0),
            "total_entries": entry_count,
            "valid_entries": valid_entries,
            "expired_entries": entry_count - valid_entries,
            "size_bytes": size_bytes,
            "database_path": str(self._db_path),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        self._logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "hit_count": getattr(self, '_hits', 0),
            "miss_count": getattr(self, '_misses', 0),
            "timestamp": datetime.utcnow().isoformat()
        }