from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text

async def retrieve_relevant_chunks(
    db: AsyncSession,
    query_embedding: list[float],
    source_types: list[str] | None = None,
    candidate_id: str | None = None,
    mandate_id: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Cosine similarity search using pgvector."""
    filters = ["1=1"]
    params = {"embedding": str(query_embedding), "limit": limit}
    
    if source_types:
        filters.append("source_type = ANY(:source_types)")
        params["source_types"] = source_types
    if candidate_id:
        filters.append("candidate_id = :candidate_id")
        params["candidate_id"] = candidate_id
    if mandate_id:
        filters.append("mandate_id = :mandate_id")
        params["mandate_id"] = mandate_id
    
    where = " AND ".join(filters)
    sql = f"""
        SELECT id, content, source_type, metadata,
               1 - (embedding <=> :embedding::vector) AS similarity
        FROM document_chunks
        WHERE {where}
        ORDER BY embedding <=> :embedding::vector
        LIMIT :limit
    """
    result = await db.execute(text(sql), params)
    rows = result.fetchall()
    return [{"id": str(r.id), "content": r.content, "source_type": r.source_type, 
             "metadata": r.metadata, "similarity": r.similarity} for r in rows]
