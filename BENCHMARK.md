# BENCHMARK.md — Lab 17 Multi-Memory Agent

## Student Information

| Field | Value |
|---|---|
| Name | Mai Tấn Thành |
| Student ID | 2A202600127 |
| Email | 26ai.thanhmt@vinuni.edu.vn |

## Setup

- Agent: LangGraph-backed multi-memory agent in `memory_agent.py`.
- Graph: real LangGraph `StateGraph` with `retrieve_memory -> generate_response` nodes.
- OpenAI: enabled; model defaults to `gpt-4o-mini-2024-07-18`.
- Memory stack: short-term buffer, JSON profile KV, JSON episodic log, Chroma semantic retrieval with keyword fallback.
- Comparison: each scenario runs once with memory disabled and once with memory enabled.
- Token efficiency: `memory_budget` uses `tiktoken` with a character-count fallback and trims recent chat first.

## Summary Metrics

| Metric | Value |
|---|---:|
| Conversations | 10 |
| Passed | 10/10 |
| Memory hit rate | 100% |
| Token budget strategy | Keep profile/episodic/semantic; trim low-priority recent chat |

## Rubric Alignment

| Rubric item | Evidence in this submission |
|---|---|
| Full memory stack | `ShortTermMemory`, `LongTermProfileMemory`, `EpisodicMemory`, `SemanticMemory` in `memory_agent.py` |
| LangGraph state/router + prompt injection | `MemoryState`, `MemoryRouter`, `StateGraph`, `retrieve_memory`, and `[USER PROFILE]`/`[EPISODIC MEMORY]`/`[SEMANTIC HITS]`/`[RECENT CONVERSATION]` prompt sections |
| Save/update + conflict handling | Profile upsert logic plus required allergy correction test |
| Benchmark | 10 multi-turn conversations below, each with no-memory vs with-memory comparison |
| Reflection | Privacy and limitations section at the end of this report |
| Bonus candidate | OpenAI JSON extraction with error fallback, Chroma semantic backend, real LangGraph flow, `tiktoken` token counting |

## Conversation Results

| # | Category | Scenario | Turns | Route | Memory hits | LLM | No-memory result | With-memory result | Pass? |
|---:|---|---|---:|---|---:|---|---|---|---|
| 1 | profile recall | Recall user name after several turns | 4 | profile | 4 | Yes | Mình không biết vì không dùng memory trong lượt này. | Tên bạn là Linh. | Pass |
| 2 | conflict update | Allergy conflict update keeps newest fact | 3 | profile | 4 | Yes | Mình không biết vì không dùng memory trong lượt này. | Profile hiện tại ghi nhận bạn dị ứng đậu nành. | Pass |
| 3 | profile recall | Preference recall | 3 | profile | 4 | Yes | Trả lời không dùng memory. | Preference hiện tại của bạn là câu trả lời ngắn, có checklist. | Pass |
| 4 | episodic recall | Episodic recall for previous debug lesson | 3 | episodic | 4 | Yes | Mình không biết vì không dùng memory trong lượt này. | Lần trước mình ghi nhận: Lesson: lần trước debug Docker, fixed bằng cách dùng docker compose service name thay vì localhost.. Kết quả: Mình đã dùng profile, episodic, semantic và recent conversation để trả lời ngắn gọn theo ngữ cảnh.. | Pass |
| 5 | semantic retrieval | Semantic retrieval for Redis | 2 | semantic | 3 | Yes | Mình chỉ có thể trả lời chung chung vì không truy xuất semantic memory. | Mình tìm thấy tài liệu liên quan: Redis stores long-term profile facts as key-value data and is useful for durable user preferences. | Pass |
| 6 | semantic retrieval | Semantic retrieval for Chroma | 2 | semantic | 3 | Yes | Mình chỉ có thể trả lời chung chung vì không truy xuất semantic memory. | Mình tìm thấy tài liệu liên quan: Chroma is a semantic vector database for retrieving relevant document chunks by meaning. | Pass |
| 7 | episodic recall | Episodic recall for completed planning task | 3 | episodic | 5 | Yes | Mình không biết vì không dùng memory trong lượt này. | Lần trước mình ghi nhận: Hoàn tất task lập kế hoạch benchmark: dùng 10 multi-turn conversations và so sánh no-memory với with-memory.. Kết quả: Task completed with a clear outcome.. | Pass |
| 8 | trim/token budget | Context budget trim still preserves profile | 7 | profile | 4 | Yes | Mình không biết vì không dùng memory trong lượt này. | Tên bạn là An. | Pass |
| 9 | trim/token budget | Token budget reporting | 3 | budget | 4 | Yes | Trả lời không dùng memory. | Context đang dùng khoảng 112/120 token; recent chat đã được trim theo priority. | Pass |
| 10 | profile recall | No hallucinated profile when memory missing | 2 | profile | 3 | Yes | Mình không biết vì không dùng memory trong lượt này. | Mình chưa có memory về dị ứng của bạn. | Pass |

## Required Conflict Test

```text
User: Tôi dị ứng sữa bò.
User: À nhầm, tôi dị ứng đậu nành chứ không phải sữa bò.
Expected profile: allergy = đậu nành
Actual with-memory answer: Profile hiện tại ghi nhận bạn dị ứng đậu nành.
```

## Privacy And Limitations

- Profile memory is the most privacy-sensitive because it can store stable PII or health-related facts such as allergies.
- A production system should support consent, TTL, and deletion across profile JSON, episodic logs, and semantic/vector stores.
- Wrong retrieval can cause the agent to answer with stale or unrelated facts, so memory hits should be surfaced with confidence and source metadata.
- Semantic retrieval uses Chroma with simple deterministic embeddings plus keyword reranking; production should use stronger embedding models.
- Token counting uses `tiktoken` when available and falls back to character-count estimation when the tokenizer is unavailable.
