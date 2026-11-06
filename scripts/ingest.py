"""批量入库 CLI。

用法：
    python -m scripts.ingest data/sample
    python -m scripts.ingest data/sample --tenant hr
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import anyio

from app.config import Settings
from app.container import build_container
from app.core.parsing import SUPPORTED_EXTENSIONS


async def run(directory: Path, tenant: str) -> int:
    settings = Settings()
    container = build_container(settings)

    files = [
        path
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        print(f"[!] {directory} 下没有可解析的文件")
        return 1

    print(f"[i] storage={settings.storage_backend} embedder={settings.embedding_provider}")
    total_chunks = 0
    started = time.perf_counter()

    for path in files:
        try:
            result = await container.ingestion.ingest(
                path.name, path.read_bytes(), tenant
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[x] {path.name}: {exc}")
            continue

        flag = "skip" if result.duplicate else "ok"
        total_chunks += 0 if result.duplicate else result.chunk_count
        print(
            f"[{flag}] {path.name}  chunks={result.chunk_count}  {result.elapsed_ms}ms"
        )

    elapsed = time.perf_counter() - started
    print(f"\n[done] {len(files)} files, {total_chunks} new chunks, {elapsed:.2f}s")
    container.store.close()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="批量导入文档到知识库")
    parser.add_argument("directory", type=Path, help="待导入的目录")
    parser.add_argument("--tenant", default="default", help="租户 id")
    args = parser.parse_args()

    if not args.directory.exists():
        print(f"[!] 目录不存在: {args.directory}")
        return 1
    return anyio.run(run, args.directory, args.tenant)


if __name__ == "__main__":
    sys.exit(main())
