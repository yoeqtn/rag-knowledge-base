from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.container import build_container
from app.core.llm import FakeChatModel
from app.main import create_app

API_KEY = "test-key"

SAMPLE_TEXT = """# 差旅报销制度

## 一、适用范围

本制度适用于公司全体正式员工。实习生参照执行，报销标准减半。

## 二、报销流程

员工出差结束后 15 个工作日内，在 OA 系统提交报销申请，附上发票原件扫描件。
直属主管审批通过后，转交财务部复核。财务部在 5 个工作日内完成打款。

## 三、住宿标准

一线城市每晚不超过 600 元，新一线城市不超过 450 元，其他城市不超过 350 元。
超标部分由个人承担。连续出差超过 7 天的，住宿标准上浮 20%。

## 四、交通标准

高铁二等座、飞机经济舱。市内交通凭票实报实销，单日上限 100 元。

## 五、特殊情况

因公产生的退票费、改签费，凭说明材料可全额报销。
"""


@pytest.fixture()
def settings() -> Settings:
    return Settings(
        api_key=API_KEY,
        storage_backend="memory",
        embedding_provider="hash",
        embedding_dim=256,
        chunk_strategy="parent_child",
        chunk_size=400,
        chunk_overlap=60,
        parent_chunk_size=1200,
        child_chunk_size=300,
        retrieval_top_k=3,
        retrieval_candidates=10,
        semantic_cache_enabled=True,
    )


@pytest.fixture()
def container(settings: Settings):
    instance = build_container(settings)
    fake = FakeChatModel()
    instance.llm = fake
    instance.qa.llm = fake
    return instance


@pytest.fixture()
def client(container):
    app = create_app(settings=container.settings, container=container)
    with TestClient(app) as test_client:
        yield test_client


def upload_sample(client: TestClient, tenant: str = "default", text: str = SAMPLE_TEXT):
    return client.post(
        "/api/documents",
        files={"file": ("差旅报销制度.md", text.encode("utf-8"), "text/markdown")},
        headers={"X-API-Key": API_KEY, "X-Tenant-Id": tenant},
    )
