from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypedDict

from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph


Role = Literal["user", "assistant"]
DEFAULT_OPENAI_MODEL = "gpt-4o-mini-2024-07-18"
_TOKEN_ENCODER: Any | None = None


class Message(TypedDict):
    role: Role
    content: str


class MemoryState(TypedDict):
    messages: list[Message]
    query: str
    user_profile: dict[str, str]
    episodes: list[dict[str, Any]]
    semantic_hits: list[str]
    recent_conversation: list[Message]
    memory_budget: int
    memory_prompt: str
    route: str
    response: str
    llm_used: bool


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "bị",
    "có",
    "của",
    "cho",
    "do",
    "em",
    "for",
    "giúp",
    "hãy",
    "i",
    "in",
    "is",
    "là",
    "me",
    "mình",
    "my",
    "of",
    "on",
    "the",
    "to",
    "tôi",
    "và",
    "với",
    "you",
}


def normalize_text(text: str) -> str:
    return text.lower().strip()


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[\wÀ-ỹ]+", normalize_text(text), flags=re.UNICODE)
        if token not in STOPWORDS and len(token) > 1
    ]


def estimate_tokens(text: str) -> int:
    global _TOKEN_ENCODER
    try:
        if _TOKEN_ENCODER is None:
            import tiktoken

            _TOKEN_ENCODER = tiktoken.encoding_for_model(DEFAULT_OPENAI_MODEL)
        return max(1, len(_TOKEN_ENCODER.encode(text)))
    except Exception:
        return max(1, len(text) // 4)


@dataclass
class ShortTermMemory:
    max_messages: int = 8
    messages: list[Message] = field(default_factory=list)

    def add(self, role: Role, content: str) -> None:
        self.messages.append({"role": role, "content": content})
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

    def retrieve(self) -> list[Message]:
        return list(self.messages)


@dataclass
class LongTermProfileMemory:
    data_path: Path
    profile: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.data_path.exists():
            self.profile = json.loads(self.data_path.read_text(encoding="utf-8"))

    def upsert(self, key: str, value: str) -> None:
        self.profile[key] = value
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.write_text(
            json.dumps(self.profile, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve(self) -> dict[str, str]:
        return dict(self.profile)


@dataclass
class EpisodicMemory:
    data_path: Path
    episodes: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.data_path.exists():
            self.episodes = json.loads(self.data_path.read_text(encoding="utf-8"))

    def append(self, summary: str, outcome: str, tags: list[str]) -> None:
        self.episodes.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "summary": summary,
                "outcome": outcome,
                "tags": tags,
            }
        )
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.data_path.write_text(
            json.dumps(self.episodes, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve(self, query: str, limit: int = 3) -> list[dict[str, Any]]:
        query_terms = set(tokenize(query))
        scored: list[tuple[int, dict[str, Any]]] = []
        for episode in self.episodes:
            text = " ".join(
                [
                    str(episode.get("summary", "")),
                    str(episode.get("outcome", "")),
                    " ".join(episode.get("tags", [])),
                ]
            )
            score = len(query_terms.intersection(tokenize(text)))
            if score:
                scored.append((score, episode))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [episode for _, episode in scored[:limit]]


@dataclass
class SemanticMemory:
    documents: list[str] = field(default_factory=list)
    collection: Any | None = None

    def __post_init__(self) -> None:
        try:
            import chromadb
            from chromadb.config import Settings

            client = chromadb.EphemeralClient(Settings(anonymized_telemetry=False))
            self.collection = client.get_or_create_collection("semantic_memory")
        except Exception:
            self.collection = None

    def add(self, document: str) -> None:
        self.documents.append(document)
        if self.collection is not None:
            try:
                self.collection.add(
                    ids=[f"doc-{len(self.documents)}"],
                    documents=[document],
                    embeddings=[self._embed(document)],
                )
            except Exception:
                self.collection = None

    def retrieve(self, query: str, limit: int = 3) -> list[str]:
        if self.collection is not None:
            try:
                result = self.collection.query(
                    query_embeddings=[self._embed(query)],
                    n_results=min(max(limit, len(self.documents)), len(self.documents)),
                )
                docs = result.get("documents", [[]])[0]
                if docs:
                    candidates = [str(doc) for doc in docs]
                    candidates.extend(doc for doc in self.documents if doc not in candidates)
                    reranked = self._keyword_rank(query, candidates)
                    return reranked[:limit]
            except Exception:
                self.collection = None

        return self._keyword_rank(query, self.documents)[:limit]

    def _keyword_rank(self, query: str, documents: list[str]) -> list[str]:
        query_terms = Counter(tokenize(query))
        scored: list[tuple[int, str]] = []
        unscored: list[str] = []
        for document in documents:
            doc_terms = Counter(tokenize(document))
            score = sum(min(count, doc_terms[term]) for term, count in query_terms.items())
            for entity in ["redis", "chroma", "docker", "faiss"]:
                if query_terms.get(entity) and doc_terms.get(entity):
                    score += 5
            if score:
                scored.append((score, document))
            else:
                unscored.append(document)
        scored.sort(key=lambda item: item[0], reverse=True)
        return [document for _, document in scored] + unscored

    def _embed(self, text: str, dims: int = 32) -> list[float]:
        vector = [0.0] * dims
        for token in tokenize(text):
            vector[hash(token) % dims] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]


class MemoryRouter:
    def route(self, query: str) -> str:
        q = normalize_text(query)
        name_intent = any(
            phrase in q
            for phrase in [
                "tên tôi",
                "tên của tôi",
                "nhớ tên",
                "my name",
                "your name for me",
            ]
        )
        if name_intent or any(word in q for word in ["dị ứng", "allergy", "sở thích", "preference"]):
            return "profile"
        if any(word in q for word in ["lần trước", "previous", "đã làm", "debug", "lesson", "outcome"]):
            return "episodic"
        if any(word in q for word in ["faq", "policy", "chunk", "tài liệu", "document", "redis", "chroma"]):
            return "semantic"
        if any(word in q for word in ["token", "budget", "trim", "context"]):
            return "budget"
        return "general"


class OpenAIMemoryClient:
    def __init__(self, model: str = DEFAULT_OPENAI_MODEL) -> None:
        load_dotenv()
        self.model = os.getenv("OPENAI_MODEL", model)
        self.enabled = bool(os.getenv("OPENAI_API_KEY"))
        self.client: Any | None = None
        if self.enabled:
            try:
                from openai import OpenAI

                self.client = OpenAI()
            except Exception:
                self.enabled = False

    def extract_profile_updates(self, user_message: str, current_profile: dict[str, str]) -> dict[str, str]:
        if not self.enabled or self.client is None:
            return {}
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                response_format={"type": "json_object"},
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Extract durable user profile facts from the latest message. "
                            "Return JSON only with shape {\"profile_updates\": {}}. "
                            "Only extract stable facts such as name, allergy, preference. "
                            "If the user corrects an older fact, keep only the newest value. "
                            "Do not extract questions as facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            {
                                "current_profile": current_profile,
                                "latest_message": user_message,
                            },
                            ensure_ascii=False,
                        ),
                    },
                ],
            )
            content = completion.choices[0].message.content or "{}"
            parsed = json.loads(content)
            updates = parsed.get("profile_updates", {})
            if isinstance(updates, dict):
                return {str(k): str(v) for k, v in updates.items() if v}
        except Exception:
            return {}
        return {}

    def generate(self, user_message: str, memory_prompt: str) -> str | None:
        if not self.enabled or self.client is None:
            return None
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                temperature=0.2,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a concise Vietnamese assistant. Use the provided memory sections "
                            "when relevant. If memory is missing, say you do not have that memory. "
                            "Do not invent profile facts."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"{memory_prompt}\n\n[USER QUESTION]\n{user_message}",
                    },
                ],
            )
            return completion.choices[0].message.content
        except Exception:
            return None


class MultiMemoryAgent:
    def __init__(
        self,
        data_dir: str | Path = "data",
        memory_budget: int = 180,
        use_llm: bool = False,
        model: str = DEFAULT_OPENAI_MODEL,
    ) -> None:
        data_path = Path(data_dir)
        self.short_term = ShortTermMemory(max_messages=8)
        self.profile = LongTermProfileMemory(data_path / "profile.json")
        self.episodic = EpisodicMemory(data_path / "episodes.json")
        self.semantic = SemanticMemory()
        self.router = MemoryRouter()
        self.memory_budget = memory_budget
        self.llm = OpenAIMemoryClient(model=model) if use_llm else None
        self.llm_calls = 0
        self.graph = self._build_graph()
        self._load_seed_semantic_docs()

    def _build_graph(self) -> Any:
        graph = StateGraph(MemoryState)
        graph.add_node("retrieve_memory", self._graph_retrieve_memory)
        graph.add_node("generate_response", self._graph_generate_response)
        graph.add_edge(START, "retrieve_memory")
        graph.add_edge("retrieve_memory", "generate_response")
        graph.add_edge("generate_response", END)
        return graph.compile()

    def _graph_retrieve_memory(self, state: MemoryState) -> MemoryState:
        return self.retrieve_memory(state["query"])

    def _graph_generate_response(self, state: MemoryState) -> MemoryState:
        state["response"] = self.generate_response(state["query"], state)
        state["llm_used"] = self.llm_calls > 0
        return state

    def _load_seed_semantic_docs(self) -> None:
        docs = [
            "Redis stores long-term profile facts as key-value data and is useful for durable user preferences.",
            "Chroma is a semantic vector database for retrieving relevant document chunks by meaning.",
            "Episodic memory records task summaries, outcomes, and lessons from previous conversations.",
            "Context window management should trim low-priority recent chat before profile and high-confidence memories.",
            "For Docker debugging, use the docker compose service name instead of localhost when one container calls another.",
        ]
        for doc in docs:
            self.semantic.add(doc)

    def receive(self, user_message: str, use_memory: bool = True) -> tuple[str, MemoryState]:
        self.short_term.add("user", user_message)

        if use_memory:
            self._save_profile_facts(user_message)
            state = self.empty_state(user_message)
            state = self.graph.invoke(state)
            response = state["response"]
            self._save_episode_if_complete(user_message, response)
        else:
            state = self.empty_state(user_message)
            response = self.generate_no_memory_response(user_message)
            state["response"] = response

        self.short_term.add("assistant", response)
        return response, state

    def empty_state(self, query: str) -> MemoryState:
        return {
            "messages": self.short_term.retrieve(),
            "query": query,
            "user_profile": {},
            "episodes": [],
            "semantic_hits": [],
            "recent_conversation": self.short_term.retrieve()[-4:],
            "memory_budget": self.memory_budget,
            "memory_prompt": "",
            "route": "none",
            "response": "",
            "llm_used": False,
        }

    def retrieve_memory(self, query: str) -> MemoryState:
        route = self.router.route(query)
        profile = self.profile.retrieve()
        episodes = self.episodic.retrieve(query)
        semantic_hits = self.semantic.retrieve(query)
        recent = self.short_term.retrieve()
        state: MemoryState = {
            "messages": recent,
            "query": query,
            "user_profile": profile,
            "episodes": episodes,
            "semantic_hits": semantic_hits,
            "recent_conversation": recent,
            "memory_budget": self.memory_budget,
            "memory_prompt": "",
            "route": route,
            "response": "",
            "llm_used": False,
        }
        return self.inject_prompt(self.trim_memory(state))

    def trim_memory(self, state: MemoryState) -> MemoryState:
        recent = list(state["recent_conversation"])
        while recent and self._state_token_estimate(state, recent) > state["memory_budget"]:
            recent.pop(0)
        state["recent_conversation"] = recent
        return state

    def _state_token_estimate(self, state: MemoryState, recent: list[Message]) -> int:
        text = json.dumps(state["user_profile"], ensure_ascii=False)
        text += json.dumps(state["episodes"], ensure_ascii=False)
        text += " ".join(state["semantic_hits"])
        text += json.dumps(recent, ensure_ascii=False)
        return estimate_tokens(text)

    def inject_prompt(self, state: MemoryState) -> MemoryState:
        profile_lines = [f"- {key}: {value}" for key, value in state["user_profile"].items()] or ["- none"]
        episode_lines = [
            f"- {episode['summary']} -> {episode['outcome']}" for episode in state["episodes"]
        ] or ["- none"]
        semantic_lines = [f"- {hit}" for hit in state["semantic_hits"]] or ["- none"]
        recent_lines = [
            f"- {message['role']}: {message['content']}" for message in state["recent_conversation"]
        ] or ["- none"]
        state["memory_prompt"] = "\n".join(
            [
                "[USER PROFILE]",
                *profile_lines,
                "[EPISODIC MEMORY]",
                *episode_lines,
                "[SEMANTIC HITS]",
                *semantic_lines,
                "[RECENT CONVERSATION]",
                *recent_lines,
            ]
        )
        return state

    def generate_response(self, user_message: str, state: MemoryState) -> str:
        if self.llm and self.llm.enabled and state["route"] == "general":
            self.llm_calls += 1
            llm_response = self.llm.generate(user_message, state["memory_prompt"])
            if llm_response:
                return llm_response
        return self._generate_rule_based_response(user_message, state)

    def _generate_rule_based_response(self, user_message: str, state: MemoryState) -> str:
        q = normalize_text(user_message)
        profile = state["user_profile"]
        name_intent = any(
            phrase in q
            for phrase in ["tên tôi", "tên của tôi", "nhớ tên", "my name", "your name for me"]
        )
        if name_intent:
            return f"Tên bạn là {profile['name']}." if "name" in profile else "Mình chưa có memory về tên bạn."
        if "dị ứng" in q or "allergy" in q:
            return (
                f"Profile hiện tại ghi nhận bạn dị ứng {profile['allergy']}."
                if "allergy" in profile
                else "Mình chưa có memory về dị ứng của bạn."
            )
        if "sở thích" in q or "preference" in q:
            return (
                f"Preference hiện tại của bạn là {profile['preference']}."
                if "preference" in profile
                else "Mình chưa có preference nào đã lưu."
            )
        if state["route"] == "episodic" and state["episodes"]:
            episode = state["episodes"][0]
            return f"Lần trước mình ghi nhận: {episode['summary']}. Kết quả: {episode['outcome']}."
        if state["route"] == "semantic" and state["semantic_hits"]:
            return f"Mình tìm thấy tài liệu liên quan: {state['semantic_hits'][0]}"
        if state["route"] == "budget":
            used = self._state_token_estimate(state, state["recent_conversation"])
            return f"Context đang dùng khoảng {used}/{state['memory_budget']} token; recent chat đã được trim theo priority."
        return "Mình đã dùng profile, episodic, semantic và recent conversation để trả lời ngắn gọn theo ngữ cảnh."

    def generate_no_memory_response(self, user_message: str) -> str:
        q = normalize_text(user_message)
        name_intent = any(
            phrase in q
            for phrase in ["tên tôi", "tên của tôi", "nhớ tên", "my name", "your name for me"]
        )
        if name_intent or any(word in q for word in ["dị ứng", "allergy", "lần trước", "previous"]):
            return "Mình không biết vì không dùng memory trong lượt này."
        if any(word in q for word in ["redis", "chroma", "docker"]):
            return "Mình chỉ có thể trả lời chung chung vì không truy xuất semantic memory."
        return "Trả lời không dùng memory."

    def _save_profile_facts(self, user_message: str) -> None:
        for key, value in self._rule_extract_profile_facts(user_message).items():
            self._upsert_profile_fact(key, value)
        if self.llm and self.llm.enabled:
            self.llm_calls += 1
            for key, value in self.llm.extract_profile_updates(user_message, self.profile.retrieve()).items():
                self._upsert_profile_fact(key, value)

    def _upsert_profile_fact(self, key: str, value: str) -> None:
        current = self.profile.retrieve()
        if key == "preference" and current.get("preference") and value not in current["preference"]:
            self.profile.upsert(key, f"{current['preference']}; {value}")
            return
        self.profile.upsert(key, value)

    def _rule_extract_profile_facts(self, user_message: str) -> dict[str, str]:
        facts: dict[str, str] = {}
        text = user_message.strip()
        lower = normalize_text(text)
        is_question = "?" in text or any(marker in lower for marker in [" gì", " không", " nhớ"])

        name_match = re.search(r"(?:tên tôi là|mình tên là|my name is)\s+([\wÀ-ỹ]+)", text, flags=re.IGNORECASE)
        if name_match:
            facts["name"] = name_match.group(1).strip()

        allergy_correction = re.search(
            r"dị ứng\s+(.+?)\s+chứ không phải\s+(.+?)(?:\.|$)",
            lower,
            flags=re.IGNORECASE,
        )
        if allergy_correction:
            facts["allergy"] = allergy_correction.group(1).strip()
        elif not is_question:
            allergy_match = re.search(r"dị ứng\s+(.+?)(?:\.|$)", lower, flags=re.IGNORECASE)
            if allergy_match:
                facts["allergy"] = allergy_match.group(1).strip()

        preference_match = re.search(
            r"(?:tôi thích|mình thích|i prefer|preference của tôi là)\s+(.+?)(?:\.|$)",
            lower,
            flags=re.IGNORECASE,
        )
        if preference_match:
            facts["preference"] = preference_match.group(1).strip()
        return facts

    def _save_episode_if_complete(self, user_message: str, response: str) -> None:
        lower = normalize_text(user_message)
        if any(word in lower for word in ["xong", "fixed", "hoàn tất", "resolved", "lesson"]):
            tags = tokenize(user_message)[:5]
            outcome = response
            if any(word in lower for word in ["xong", "hoàn tất", "resolved"]):
                outcome = "Task completed with a clear outcome."
            elif len(outcome) > 180:
                outcome = outcome[:177].rstrip() + "..."
            self.episodic.append(
                summary=user_message,
                outcome=outcome,
                tags=tags,
            )
