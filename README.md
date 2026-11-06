# RAG 知识库问答系统

一个面向企业文档的检索增强问答服务：上传 PDF / Word / Markdown，基于混合检索生成带**原文引用**的回答，支持流式输出、多轮对话、语义缓存和成本统计。

> 目标不是"调通一个 LangChain Demo"，而是把 RAG 当成一个**后端系统**来做：  
> 分层清晰、可替换、可评测、可观测。

---

## 功能

| 能力   | 说明                                          |
| ---- | ------------------------------------------- |
| 文档解析 | PDF / DOCX / Markdown / TXT，兼容 UTF-8 与 GBK  |
| 切块策略 | 定长、递归、**父子块**三种，可配置                         |
| 向量化  | 任何 OpenAI 兼容的 `/embeddings` 接口，带 LRU 缓存     |
| 混合检索 | 向量召回 + BM25 召回，**RRF 融合**                   |
| 重排序  | 可选 cross-encoder rerank（硅基流动 / 阿里云百炼 / Jina） |
| 父块回溯 | 子块精确召回，父块补全上下文再喂给 LLM                       |
| 引用溯源 | 回答中的 `[1]` 映射回文件名、块号和原文片段                   |
| 流式输出 | SSE，先推检索结果再推 token                          |
| 多轮对话 | 自动把追问改写成独立问题再检索                             |
| 语义缓存 | 相似问题命中缓存，省时省钱                               |
| 可观测  | 每次问答记录 token、成本、延迟、缓存命中                     |
| 双存储  | 内存（测试）/ PostgreSQL + pgvector（生产）           |

---

## 架构

```
                    ┌───────────────────────────────────────────┐
   上传文档          │              FastAPI (app/api)             │
  ─────────────────►│  /api/documents   /api/chat   /api/stats  │
                    └───────────────┬───────────────────────────┘
                                    │
              ┌─────────────────────┴─────────────────────┐
              │                                           │
     ┌────────▼─────────┐                       ┌─────────▼────────┐
     │ IngestionService │                       │    QAService     │
     │ parse            │                       │ rewrite (多轮)    │
     │ chunk            │                       │ embed            │
     │ embed            │                       │ retrieve         │
     │ store            │                       │ cache            │
     └────────┬─────────┘                       │ prompt + LLM     │
              │                                 │ citations        │
              │                                 │ log + cost       │
              │                                 └─────────┬────────┘
              │                                           │
     ┌────────▼───────────────────────────────────────────▼────────┐
     │                     Store (Protocol)                        │
     │   MemoryStore（测试）        PgStore（PostgreSQL + pgvector） │
     └─────────────────────────────────────────────────────────────┘

     检索链路：query ─┬─► 向量召回 top-30 ─┐
                     │                    ├─► RRF 融合 ─► rerank(可选) ─► 父块回溯 ─► top-5
                     └─► BM25 召回 top-30 ┘
```

**为什么检索要走两路**：向量擅长语义改写（"怎么报销" ≈ "报销流程"），但对型号、编号、专有名词不敏感；BM25 正好相反。两路 RRF 融合，比单路稳定得多。

---

## 快速开始

### 1. 安装

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2. 零依赖跑通（内存存储 + 假向量）

```bash
python -m uvicorn app.main:app --reload
```

打开 <http://127.0.0.1:8000/docs> 。

> 默认 `EMBEDDING_PROVIDER=hash`，只用于跑通链路，**语义检索效果很差**。  
> 正式使用前必须换成真实 embedding，见下一步。

### 3. 配置真实模型

```bash
copy .env.example .env          # Windows
```

编辑 `.env`：

```ini
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DIM=1024
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=sk-xxx

LLM_BASE_URL=https://api.deepseek.com/v1
LLM_API_KEY=sk-xxx
LLM_MODEL=deepseek-chat
```

任何 OpenAI 兼容服务都可以：DeepSeek、通义、智谱、硅基流动、OpenAI。

**重排序（可选）** 默认关闭。不同厂商的 rerank 接口格式不一样，项目里做了适配：

```ini
# 硅基流动
RERANK_ENABLED=true
RERANK_PROVIDER=siliconflow
RERANK_MODEL=BAAI/bge-reranker-v2-m3
RERANK_BASE_URL=https://api.siliconflow.cn/v1

# 阿里云百炼（模型名以控制台为准，常见 gte-rerank-v2）
RERANK_ENABLED=true
RERANK_PROVIDER=dashscope
RERANK_MODEL=gte-rerank-v2
RERANK_BASE_URL=https://dashscope.aliyuncs.com/api/v1
```

> 百炼的接口是嵌套的 `input/parameters`，结果在 `output.results`；硅基流动是扁平的。
> 代码按 `RERANK_PROVIDER` 自动切换。如果返回 404，把 `RERANK_BASE_URL` 直接填成
> 完整地址（只要路径里含 `services/rerank` 就会原样使用）。

### 4. 用 PostgreSQL + pgvector

```bash
docker compose up -d db
```

`.env` 中设置：

```ini
STORAGE_BACKEND=postgres
DATABASE_URL=postgresql+psycopg://rag:rag@localhost:5433/rag
```

首次启动自动建表、建 pgvector 扩展和 HNSW 索引。

### 5. 导入样例文档并提问

```bash
python -m scripts.ingest data/sample

curl -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\": \"成都的住宿标准是多少？\"}"
```

---

## API

| 方法     | 路径                    | 说明                            |
| ------ | --------------------- | ----------------------------- |
| POST   | `/api/documents`      | 上传文档（multipart，需 `X-API-Key`） |
| GET    | `/api/documents`      | 文档列表                          |
| DELETE | `/api/documents/{id}` | 删除文档及其所有块                     |
| POST   | `/api/chat`           | 同步问答                          |
| POST   | `/api/chat/stream`    | SSE 流式问答                      |
| GET    | `/api/health`         | 存储 / 模型 / 缓存状态                |
| GET    | `/api/stats`          | 文档数、问答数、token、成本、缓存命中率        |

请求头：

- `X-API-Key`：写操作鉴权，值取自 `.env` 的 `API_KEY`
- `X-Tenant-Id`：租户隔离，默认 `default`

### 流式事件

```
data: {"type": "sources", "sources": [...]}     # 先告诉前端检索到了什么
data: {"type": "token", "text": "根据"}
data: {"type": "token", "text": "制度，"}
data: {"type": "done", "answer": "...", "citations": [...], "usage": {...}}
```

---

## 评测

```bash
# 只评检索，不花钱
python -m scripts.evaluate --docs data/sample --offline --top-k 3

# 完整评测（需要 LLM_API_KEY）
python -m scripts.evaluate --docs data/sample --top-k 3
```

数据集格式（`eval/dataset.example.jsonl`）：

```json
{"question": "成都的住宿标准是多少钱一晚？", "expected_source": "差旅报销制度.md", "expected_keywords": ["450"]}
```

指标：`source_hit@k`、`MRR`、关键词命中率、平均延迟、总成本。

### 切块策略对比实验

```bash
python -m scripts.experiment_chunking --dataset eval/dataset.hard.jsonl --top-k 3
```

同一批文档、同一批问题，只改切块配置，输出可直接贴进简历的对比表。

实测环境：11 篇制度文档、17 条口语化问题、Qwen embedding、top_k=3。

| 配置 | 块数 | Recall@3 | MRR | 上下文命中率 | 检索精度 | 平均上下文 |
|---|---:|---:|---:|---:|---:|---:|
| fixed-300 | 32 | 100% | 1.0000 | 100% | 45.10% | 797 字 |
| recursive-200 | 59 | 100% | 0.9706 | 100% | 49.02% | 459 字 |
| recursive-300 | 33 | 100% | 0.9314 | 94.12% | 41.18% | 684 字 |
| recursive-500 | 19 | 100% | 0.9706 | 94.12% | 43.14% | 1049 字 |
| parent-200 | 49 | 100% | 1.0000 | 100% | **74.51%** | 954 字 |
| parent-300 | 30 | 100% | 1.0000 | 100% | 62.75% | 1053 字 |
| parent-500 | 19 | 100% | 0.9706 | 100% | 44.12% | 1479 字 |

**结论**：实验特意让 recursive 和 parent 在**相同块长**下对比，避免把"块变小"的收益
误算到"父子块"头上。

- 块长 200：49.0% → 74.5%，父子块 +25.5 个百分点
- 块长 300：41.2% → 62.8%，+21.6 个百分点
- 块长 500：43.1% → 44.1%，基本持平

也就是说，父子块的收益主要来自**用小粒度子块做检索**；块本身够大时，父块回溯没什么增量。
这个结论比"父子块更好"有用得多——面试官会接着问"那什么时候不该用"，你已经有答案了。

⚠️ 两个必须说清楚的前提：

1. 语料只有 11 篇、150 个块以内，**这组数字是方向性的，不是定论**。
   要写进简历，请把语料扩到 50 篇以上、问题扩到 100 条以上再跑一次。
2. 自带的 8 条 `dataset.example.jsonl` 在任何配置下都是 100%——题目直接抄了原文，
   没有任何区分度，**不要拿它当实验结果**。

### 召回方式对比实验

```bash
python -m scripts.experiment_retrieval --dataset eval/dataset.hard.jsonl --top-k 3
```

固定切块配置（parent_child / 子块 300 字），只改检索链路：

| 召回方式 | Recall@3 | MRR | 上下文命中率 | 检索精度 | 检索延迟 |
|---|---:|---:|---:|---:|---:|
| vector-only | 100% | 0.9706 | 100% | 66.67% | 56.3ms |
| bm25-only | 94.12% | 0.7941 | 100% | 53.92% | **8.3ms** |
| hybrid | 100% | **1.0000** | 100% | 62.75% | 60.1ms |

**结论**：

- 纯 BM25 最快（8ms vs 56ms），但会在"笔记本丢了"这类**没有词面重叠**的问题上漏召回
- 纯向量几乎全能召回，但有一道题的答案文档排到了第 2
- 混合召回把 MRR 拉到 1.0，代价是延迟回到向量水平

⚠️ 17 道题里 1 道题就是 5.9 个百分点。精度 66.67% vs 62.75% 的差距小于单题波动，
**不能据此说"纯向量更准"**。这张表能支撑的结论只有 Recall 和 MRR 的方向。

报告里会自动列出「分歧题」——只有这些题才能说明两路的差别。

---

## 项目结构

```
app/
├── main.py                 # 应用入口
├── config.py               # 全部可调参数
├── container.py            # 依赖装配（唯一 new 具体实现的地方）
├── schemas.py              # 请求/响应模型
├── api/
│   ├── deps.py             # 鉴权、租户、容器依赖
│   ├── routes_documents.py
│   ├── routes_chat.py
│   └── routes_system.py
├── core/
│   ├── parsing.py          # 文件解析
│   ├── chunking.py         # 三种切块策略
│   ├── embedding.py        # Embedder 协议 + API/Hash 实现 + 缓存
│   ├── bm25.py             # 纯 Python BM25 + jieba
│   ├── fusion.py           # RRF
│   ├── rerank.py           # 重排序
│   ├── stores.py           # 存储协议 + 内存/PG 实现
│   ├── retrieval.py        # 混合检索主流程
│   ├── llm.py              # 模型调用 + 流式
│   ├── prompt.py           # Prompt 模板
│   ├── citations.py        # 引用解析
│   └── cache.py            # 语义缓存
└── services/
    ├── ingest.py           # 入库编排
    └── qa.py               # 问答编排

scripts/
├── ingest.py               # 批量导入
├── evaluate.py             # 检索/问答评测
├── experiment_chunking.py  # 切块策略对比实验
└── smoke_test.py           # 端到端冒烟测试

data/sample/                # 11 篇示例制度文档
eval/
├── dataset.example.jsonl   # 8 条基础样例（无区分度，仅演示格式）
└── dataset.hard.jsonl      # 17 条口语化问题（用于实验）

docs/
├── 01-项目导读.md           # 读代码的顺序与重点
├── 02-简历写法.md           # 怎么把这个项目写进简历
└── 03-面试备战.md           # 高频问题与回答骨架
```

---

## 设计决策（面试重点）

1. **为什么父块/子块分离**  
   子块小 → 检索精度高；但上下文太碎，LLM 容易答不全。命中子块后回溯父块，兼顾两者。代价是存储翻倍、需要额外一次查询。
2. **为什么用 RRF 而不是分数加权**  
   余弦相似度在 0~1，BM25 分数是几十，量纲不同，人工调权重很脆弱。RRF 只看排名，`1/(60+rank)` 求和，几乎不用调参。
3. **为什么自己写 BM25 而不是用 Elasticsearch**  
   单机几千到几万块数据，纯 Python BM25 是毫秒级，省一个中间件。数据量再大再换 ES。
4. **为什么存储要抽象**  
   测试用内存实现，毫秒级、零依赖；生产用 pgvector。上层业务代码完全不需要改。这是依赖倒置最直接的例子。
5. **为什么同步 SQLAlchemy + 异步 LLM**  
   数据库查询都是短操作，用 `anyio.to_thread` 放到线程池；LLM 流式是长连接 I/O，必须异步。混用的时候要清楚边界在哪。
6. **为什么回答必须带引用**  
   RAG 最大的风险是幻觉。引用让用户能核对原文，也让评测有据可依。

---

## 已知限制 / Roadmap

- [ ] 扫描版 PDF 需要 OCR（当前只处理文本层）
- [ ] 引用只到块级别，未定位到页码
- [ ] BM25 索引在每个进程内存里，多 worker 会各建一份
- [ ] 未做增量重建索引，删文档后靠版本号失效
- [ ] 没有前端（可用 `/docs` 或 Streamlit 快速补）
- [ ] 没有权限体系，只有租户隔离
- [ ] 未接入 LangChain/LangGraph —— 刻意手写，方便讲清每一步

---

## 测试

```bash
pytest
```

覆盖：切块策略、BM25 排序、RRF 融合、引用解析、上传/问答/流式/删除接口、父块回溯。

## License

MIT
