"""文件解析：把上传的原始字节变成纯文本。"""

from __future__ import annotations

import io
from pathlib import Path

SUPPORTED_EXTENSIONS = {".txt", ".md", ".markdown", ".pdf", ".docx"}


class UnsupportedFileType(ValueError):
    pass


class EmptyDocument(ValueError):
    pass


def parse_bytes(filename: str, data: bytes) -> str:
    """按扩展名分发到具体解析器，返回清洗后的纯文本。"""
    suffix = Path(filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise UnsupportedFileType(
            f"不支持的文件类型: {suffix or '(无扩展名)'}，"
            f"支持: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )

    if suffix == ".pdf":
        text = _parse_pdf(data)
    elif suffix == ".docx":
        text = _parse_docx(data)
    else:
        text = _decode_text(data)

    text = normalize_text(text)
    if not text.strip():
        raise EmptyDocument("文件没有可提取的文本内容（可能是扫描版 PDF）")
    return text


def _decode_text(data: bytes) -> str:
    """中文环境常见编码都试一遍。Windows 上的 txt 经常是 GBK。"""
    for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def _parse_pdf(data: bytes) -> str:
    from pypdf import PdfReader

    reader = PdfReader(io.BytesIO(data))
    pages: list[str] = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n\n".join(pages)


def _parse_docx(data: bytes) -> str:
    import docx

    document = docx.Document(io.BytesIO(data))
    blocks: list[str] = []
    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            blocks.append(paragraph.text)

    # 表格里的内容也是知识，别漏
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                blocks.append(" | ".join(cells))

    return "\n\n".join(blocks)


def normalize_text(text: str) -> str:
    """统一换行、压掉过多空行，避免切块时出现大量空块。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace(" ", " ").replace("　", " ")
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)

    while "\n\n\n" in text:
        text = text.replace("\n\n\n", "\n\n")
    return text.strip()
