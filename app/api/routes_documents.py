"""文档管理接口。"""

from __future__ import annotations

from anyio import to_thread
from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile

from app.api.deps import get_container, get_tenant, require_api_key
from app.container import Container
from app.core.parsing import EmptyDocument, UnsupportedFileType
from app.schemas import DocumentListResponse, DocumentOut, IngestResponse

router = APIRouter(prefix="/api/documents", tags=["documents"])


@router.post("", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def upload_document(
    file: UploadFile = File(...),
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> IngestResponse:
    data = await file.read()
    try:
        result = await container.ingestion.ingest(file.filename or "unnamed", data, tenant)
    except (UnsupportedFileType, EmptyDocument, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return IngestResponse(
        document=_to_out(result.document),
        duplicate=result.duplicate,
        chunk_count=result.chunk_count,
        elapsed_ms=result.elapsed_ms,
    )

@router.get("", response_model=DocumentListResponse)
async def list_documents(
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
) -> DocumentListResponse:
    items = await to_thread.run_sync(
        container.store.list_documents, tenant, limit, offset
    )
    return DocumentListResponse(
        total=len(items), items=[_to_out(item) for item in items]
    )

@router.delete("/{document_id}", dependencies=[Depends(require_api_key)])
async def delete_document(
    document_id: str,
    tenant: str = Depends(get_tenant),
    container: Container = Depends(get_container),
) -> dict:
    deleted = await to_thread.run_sync(
        container.store.delete_document, document_id, tenant
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="文档不存在")
    return {"deleted": True, "document_id": document_id}

def _to_out(document) -> DocumentOut:
    return DocumentOut(
        id=document.id,
        filename=document.filename,
        size_bytes=document.size_bytes,
        chunk_count=document.chunk_count,
        created_at=document.created_at,
    )
