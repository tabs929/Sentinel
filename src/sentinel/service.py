"""High-level orchestration used by both the API and the CLI.

Owns the wiring between the :class:`Store`, the :class:`ToolRegistry`, the
LLM client and the :class:`Agent`, so callers get a single, simple surface.
"""

from __future__ import annotations

from sentinel.agent.loop import Agent, TurnResult
from sentinel.config import Settings, get_settings
from sentinel.llm.client import AnthropicClient, LLMClient
from sentinel.models import Session, Trace, Turn
from sentinel.observability.tracing import Listener
from sentinel.state.store import Store
from sentinel.tools.base import ToolRegistry, build_default_registry


class SessionNotFoundError(Exception):
    pass


class SentinelService:
    def __init__(
        self,
        store: Store,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> None:
        self.store = store
        self.settings = settings or get_settings()
        self.registry = registry or build_default_registry(self.settings)
        self._llm = llm

    @classmethod
    async def create(
        cls,
        *,
        settings: Settings | None = None,
        llm: LLMClient | None = None,
        registry: ToolRegistry | None = None,
    ) -> SentinelService:
        settings = settings or get_settings()
        store = await Store.open(settings)
        return cls(store, settings=settings, llm=llm, registry=registry)

    async def close(self) -> None:
        await self.store.close()

    # --- LLM (lazy) ------------------------------------------------------------

    def _get_llm(self) -> LLMClient:
        if self._llm is None:
            self._llm = AnthropicClient(self.settings)
        return self._llm

    def _build_agent(self, listeners: list[Listener] | None = None) -> Agent:
        return Agent(
            self._get_llm(),
            self.registry,
            store=self.store,
            settings=self.settings,
            listeners=listeners,
        )

    # --- sessions --------------------------------------------------------------

    async def start_session(self, title: str | None = None) -> Session:
        session = Session(title=title)
        await self.store.save_session(session)
        return session

    async def send_message(
        self,
        session_id: str,
        message: str,
        *,
        listeners: list[Listener] | None = None,
    ) -> TurnResult:
        session = await self.store.get_session(session_id)
        if session is None:
            raise SessionNotFoundError(session_id)
        agent = self._build_agent(listeners=listeners)
        return await agent.run_turn(session, message)

    async def get_session(self, session_id: str) -> Session | None:
        return await self.store.get_session(session_id)

    async def list_sessions(self) -> list[Session]:
        return await self.store.list_sessions()

    async def get_turns(self, session_id: str) -> list[Turn]:
        return await self.store.get_turns(session_id)

    async def get_turn(self, session_id: str, index: int) -> Turn | None:
        return await self.store.get_turn(session_id, index)

    # --- traces ----------------------------------------------------------------

    async def list_traces(self) -> list[Trace]:
        return await self.store.list_traces()

    async def get_trace(self, trace_id: str) -> Trace | None:
        return await self.store.get_trace(trace_id)

    async def get_trace_for_turn(self, session_id: str, turn_index: int) -> Trace | None:
        return await self.store.get_trace_for_turn(session_id, turn_index)
