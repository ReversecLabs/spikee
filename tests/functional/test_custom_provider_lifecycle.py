"""Local HTTP regressions for repeated OpenAI-compatible provider calls."""

import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from threading import Thread
from unittest.mock import AsyncMock

import pytest

from spikee.providers.azure_openai import AnyLLMAzureOpenAIProvider
from spikee.providers.custom import AnyLLMCustomProvider
from spikee.providers.openai import AnyLLMOpenAIProvider
from spikee.providers.openrouter import AnyLLMOpenRouterProvider
from spikee.templates.provider import ProviderError
from spikee.utilities.llm_message import HumanMessage


@pytest.fixture
def endpoint(monkeypatch):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            tier = request["messages"][0]["content"]
            payload = json.dumps(
                {
                    "id": "fixture",
                    "object": "chat.completion",
                    "created": 0,
                    "model": request["model"],
                    "service_tier": tier,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "fixture reply",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 2,
                        "completion_tokens": 3,
                        "total_tokens": 5,
                    },
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{server.server_port}/v1"
    monkeypatch.setenv("CUSTOM_API_URL", url)
    monkeypatch.setenv("CUSTOM_API_KEY", "fixture")
    monkeypatch.setenv("OPENROUTER_API_KEY", "fixture")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("OPENAI_BASE_URL", url)
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", "fixture")
    monkeypatch.setenv("AZURE_OPENAI_ENDPOINT", url)
    monkeypatch.delenv("AZURE_OPENAI_API_VERSION", raising=False)
    monkeypatch.delenv("OPENAI_API_VERSION", raising=False)
    monkeypatch.setattr(
        AnyLLMOpenRouterProvider, "base_url", property(lambda self: url)
    )
    yield
    server.shutdown()
    thread.join()
    server.server_close()


@pytest.mark.parametrize(
    "provider_class",
    [
        AnyLLMCustomProvider,
        AnyLLMOpenRouterProvider,
        AnyLLMAzureOpenAIProvider,
        AnyLLMOpenAIProvider,
    ],
)
def test_repeated_calls_close_clients_on_their_own_loop(
    endpoint, provider_class, monkeypatch
):
    provider = provider_class()
    provider.setup("fixture", max_tokens=12, temperature=0.2, timeout=7)
    create = provider.create_client
    closed = []

    def tracked_client():
        llm = create()
        assert llm.client.timeout == 7
        close = llm.client.close
        loop = asyncio.get_running_loop()

        async def tracked_close():
            assert asyncio.get_running_loop() is loop
            await close()
            assert llm.client.is_closed()
            closed.append(loop)

        llm.client.close = tracked_close
        return llm

    monkeypatch.setattr(provider, "create_client", tracked_client)
    for _ in range(3):
        reply = provider.invoke([HumanMessage("default")])
        assert reply.content == "fixture reply"
        assert reply.original_response.usage.total_tokens == 5
    assert len(closed) == 3
    assert all(loop.is_closed() for loop in closed)


def test_shared_provider_calls_use_independent_clients(endpoint):
    provider = AnyLLMOpenRouterProvider()
    provider.setup("fixture")
    with ThreadPoolExecutor(max_workers=2) as pool:
        replies = list(
            pool.map(lambda _: provider.invoke([HumanMessage("default")]), range(4))
        )
    assert [r.content for r in replies] == ["fixture reply"] * 4


def test_client_closes_when_completion_fails(monkeypatch):
    provider = AnyLLMCustomProvider()
    monkeypatch.setenv("CUSTOM_API_URL", "http://localhost.invalid/v1")
    monkeypatch.setenv("CUSTOM_API_KEY", "fixture")
    provider.setup("fixture")
    client = AsyncMock()
    client.acompletion.side_effect = ValueError("fixture failure")
    monkeypatch.setattr(provider, "create_client", lambda: client)
    with pytest.raises(ProviderError, match="fixture failure"):
        provider.invoke([HumanMessage("default")])
    client.client.close.assert_awaited_once()
