# Spikee Target Implementation Guide

Targets are the systems under test. They live in `targets/` as Python files exposing a class that inherits from one of the base classes below. Spikee sends attack payloads to `process_input()` and judges the response.

---

## 1. Naming Conventions

- Filename: `snake_case.py` — `{application}_{feature}.py`
- Class name: `PascalCase` — `{Application}{Feature}Target`
- **Always confirm the name with the user** — offer 2–3 options, let them choose or provide their own.

| Example | Filename | Class |
|---|---|---|
| Acme title generation | `acme_title.py` | `AcmeTitleTarget` |
| Example chatbot | `example_chatbot.py` | `ExampleChatbot` |
| AWS Bedrock guardrail | `aws_bedrock_guardrail.py` | `AWSBedrockGuardrailTarget` |

The feature suffix can be omitted for a single-endpoint target (`acme.py` / `AcmeTarget`), but add it as soon as a second endpoint is introduced.

---

## 2. Information-Gathering

**Always ask for a captured HTTP request + response first** (Burp Suite / DevTools). It answers most questions below in one step.

> "Can you paste a captured HTTP request and response?"

If not available, ask:

| # | Question | Drives |
|---|---|---|
| 2.1 | Single-turn (each message independent) or multi-turn (conversation history)? | Base class (Section 3) |
| 2.2 | HTTP REST, WebSocket, or both? | Transport pattern (Sections 7, 8) |
| 2.3 | Where in the response body is the reply text? | Response parsing |
| 2.4 | How does it authenticate? (API key / cookie / OAuth2 / IMDS / GCP ADC / JWT) | Auth pattern (Section 4) |
| 2.5 | Session/thread/conversation ID — how is a new one created? | Multi-turn session management |
| 2.6 | Any non-standard headers? (`X-Project-Id`, `Auth0-Client`, `x-*`, `Origin`, etc.) | Request headers |
| 2.7 | How does it signal a block? (HTTP status, JSON flag, specific reply string) | `GuardrailTrigger` (Section 5) |
| 2.8 | Runtime variants to test? (model, env, guardrail on/off) | `target_options` |
| 2.9 | Route through intercepting proxy? | `proxy=` option |

**Base class selection** — decide before writing:

| Scenario | Base class | Import |
|---|---|---|
| Single-turn LLM or guardrail | `Target` | `from spikee.templates.target import Target` |
| Multi-turn, spikee-managed history | `SimpleMultiTarget` | `from spikee.templates.simple_multi_target import SimpleMultiTarget` |
| Multi-turn, custom/complex state | `MultiTarget` | `from spikee.templates.multi_target import MultiTarget` |

> Prefer `SimpleMultiTarget` for most chatbots — it provides session-ID mapping and history helpers out of the box.

---

## 3. Required Methods

Every target must implement:

```python
def get_description(self) -> ModuleDescriptionHint:
    return [ModuleTag.LLM], "Brief human-readable description."

def get_available_option_values(self) -> ModuleOptionsHint:
    # (List[str], needs_llm_provider: bool). First item = default.
    return ["mode=default", "mode=strict"], False  # or ([], False) if no options

def process_input(
    self,
    input_text: str,
    system_message: Optional[str] = None,
    target_options: Optional[str] = None,
    # Multi-turn only:
    # spikee_session_id: Optional[str] = None,
    # backtrack: Optional[bool] = False,
) -> TargetResponseHint:
    ...
```

**Return type:** `str` (LLM reply) · `True` (guardrail bypassed) · `False` (guardrail blocked) · `(str, Any)` (reply + ignored metadata)

**Module tags** (use in `get_description()`): `ModuleTag.LLM`, `ModuleTag.MULTI`, `ModuleTag.SINGLE`, `ModuleTag.IMAGE`, `ModuleTag.AUDIO`

---

## 4. Authentication Patterns

### 4.1 Static API Key / Bearer Token

```python
import os
from dotenv import load_dotenv
load_dotenv()

API_KEY = os.getenv("MY_API_KEY")  # preferred; or hardcode for testing
headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
```

### 4.2 Static Session Cookie

```python
SESSION_COOKIE = "eyJpc..."  # grab from browser after login
headers = {"Cookie": f"session={SESSION_COOKIE}", "Content-Type": "application/json"}
```

### 4.3 Username / Password (Form Login)

Authenticate once in `__init__`, store cookies, reuse on every request.

```python
def get_auth(self) -> None:
    response = requests.post(
        "https://example.com/login",
        data={"username": os.getenv("TARGET_USERNAME"), "password": os.getenv("TARGET_PASSWORD")},
        allow_redirects=False,
    )
    if response.status_code != 302:
        raise Exception(f"Auth failed: {response.status_code}")
    self._cookies = response.cookies
```

### 4.4 Azure Managed Identity (IMDS)

Requires the test machine to be an Azure VM with an assigned managed identity.

```python
def __init__(self):
    super().__init__()
    try:
        self.__access_token = requests.get(
            "http://169.254.169.254/metadata/identity/oauth2/token"
            "?api-version=2018-02-01&resource=https%3A%2F%2Fmanagement.azure.com%2F",
            headers={"Metadata": "true"},
        ).json()["access_token"]
    except Exception as e:
        print("ERROR: Failed to collect Azure access_token:", e)
        exit(1)
```

### 4.5 GCP Application Default Credentials

Requires `google-auth` and `gcloud auth application-default login` (or `GOOGLE_APPLICATION_CREDENTIALS`).

```python
import google.auth, google.auth.transport.requests

def __get_gcp_token(self) -> None:
    creds, _ = google.auth.default()
    creds.refresh(google.auth.transport.requests.Request())
    self.__gcp_token = creds.token
```

Call in `__init__`, refresh on `401`.

### 4.6 Auth0 / OAuth2 Refresh Token

Short-lived access tokens (~300 s), long-lived refresh token hardcoded or in `.env`. Extract from a captured auth request: token endpoint, `client_id`, any custom headers (`Auth0-Client`, `Origin`), and the initial `refresh_token`.

> **Auth0 rotates the refresh token on every use** — always capture `refresh_token` from the response or the next call will fail.

```python
import time

AUTH0_TOKEN_URL  = "https://auth.example.com/oauth/token"
AUTH0_CLIENT_ID  = "yourClientId"
AUTH0_CLIENT_HDR = "eyJ..."      # Auth0-Client header value
REFRESH_TOKEN    = "v1.initial"  # update when it rotates

class MyTarget(Target):
    def __init__(self):
        super().__init__()
        self._access_token: Optional[str] = None
        self._refresh_token: str = REFRESH_TOKEN
        self._token_expiry: float = 0.0
        self._refresh_access_token()  # fail fast on bad credentials

    def _refresh_access_token(self) -> None:
        response = requests.post(
            AUTH0_TOKEN_URL,
            headers={"Content-Type": "application/x-www-form-urlencoded",
                     "Auth0-Client": AUTH0_CLIENT_HDR,
                     "Origin": "https://app.example.com"},
            data={"client_id": AUTH0_CLIENT_ID, "grant_type": "refresh_token",
                  "redirect_uri": "https://app.example.com",
                  "refresh_token": self._refresh_token},
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        self._access_token = body["access_token"]
        self._token_expiry = time.time() + body.get("expires_in", 300) - 15
        if "refresh_token" in body:        # capture rotated token
            self._refresh_token = body["refresh_token"]

    def _ensure_token(self) -> str:
        if self._access_token is None or time.time() >= self._token_expiry:
            self._refresh_access_token()
        return self._access_token
```

### 4.7 Internal JWT (Self-Fetched)

When the target's own auth service issues JWTs, fetch lazily and validate expiry with `PyJWT`.

```python
import jwt

def __check_jwt(self) -> bool:
    try:
        jwt.decode(self.__jwt, leeway=0.5,
                   options={"verify_signature": False, "verify_exp": True},
                   algorithms=["RS256"])
        return True
    except Exception:
        return False

def __generate_jwt(self) -> None:
    self.__jwt = requests.get("https://auth.example.com/token",
                               timeout=10, verify=False).content.decode()

def send_message(self, text: str) -> str:
    if self.__jwt is None or not self.__check_jwt():
        self.__generate_jwt()
    headers = {"Authorization": f"Bearer {self.__jwt}", ...}
```

---

## 5. Guardrail Handling

**Guardrail targets** return `bool` instead of a string — `True` = attack bypassed, `False` = blocked.

```python
def process_input(self, input_text, system_message=None, target_options=None) -> bool:
    result = response.json()
    return not result.get("attack_detected", False)  # True = bypassed
```

**LLM targets** raise exceptions to signal blocks:

```python
raise GuardrailTrigger("Reason", categories={"policy": True})  # blocked by app
raise RetryableError("Rate limited", retry_period=60)           # 429 / transient
```

Common detection signals:

| Signal | Pattern |
|---|---|
| HTTP 400 | `if e.response.status_code == 400: raise GuardrailTrigger(...)` |
| HTTP 403 (WAF) | `if e.response.status_code == 403: raise GuardrailTrigger("WAF", categories={"waf": True})` |
| JSON flag | `if result.get("blocked"): raise GuardrailTrigger(...)` |
| Reply string | `if reply == "That can't be answered here.": raise GuardrailTrigger(...)` |
| API action field | `result = response.json(); if result["action"] == "GUARDRAIL_INTERVENED": return True` |

**Reactive 401 retry** (short-lived tokens):

```python
response = requests.post(url, headers=headers, json=payload, timeout=30)
if response.status_code == 401:
    self._refresh_access_token()
    headers["Authorization"] = f"Bearer {self._access_token}"
    response = requests.post(url, headers=headers, json=payload, timeout=30)
response.raise_for_status()
```

---

## 6. Single-Turn Template

```python
from typing import Optional
import os, requests
from dotenv import load_dotenv
from spikee.templates.target import Target
from spikee.tester import GuardrailTrigger
from spikee.utilities.enums import ModuleTag
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, TargetResponseHint
from spikee.utilities.modules import parse_options
from urllib3.exceptions import InsecureRequestWarning

load_dotenv()
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

API_KEY = os.getenv("MY_API_KEY", "")
API_URL = "https://example.com/api/chat"


class MyTarget(Target):
    def __init__(self):
        super().__init__()

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM], "Brief description."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return ["", "proxy=localhost:8080"], False

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
    ) -> TargetResponseHint:
        opts = parse_options(target_options)
        proxy_host = opts.get("proxy")
        proxies = {"http": f"http://{proxy_host}", "https": f"http://{proxy_host}"} if proxy_host else {}

        try:
            response = requests.post(
                API_URL,
                headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
                json={"message": input_text},
                proxies=proxies, verify=not bool(proxy_host), timeout=30,
            )
            response.raise_for_status()
            return response.json().get("answer", "")

        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 400:
                    raise GuardrailTrigger(f"Guardrail triggered: {e}")
            print(f"HTTP error: {e}")
            raise


if __name__ == "__main__":
    target = MyTarget()
    print(target.process_input("Hello!"))
```

---

## 7. Multi-Turn Template

`SimpleMultiTarget` helpers:
- `_get_id_map(sid)` / `_update_id_map(sid, value)` — maps Spikee session IDs → target session IDs
- `_get_conversation_data(sid)` / `_update_conversation_data(sid, data)` — history list
- `_append_conversation_data(sid, role, content)` — append one turn

For complex/custom state use `MultiTarget` directly and call `_get_target_data(sid)` / `_update_target_data(sid, data)` with an arbitrary dict.

```python
import uuid
from typing import Optional
import requests
from dotenv import load_dotenv
from spikee.templates.simple_multi_target import SimpleMultiTarget
from spikee.tester import GuardrailTrigger
from spikee.utilities.enums import ModuleTag, Turn
from spikee.utilities.hinting import ModuleDescriptionHint, ModuleOptionsHint, TargetResponseHint
from spikee.utilities.modules import parse_options

load_dotenv()
API_URL = "https://example.com/api/chat"


class MyChatbotTarget(SimpleMultiTarget):
    def __init__(self):
        super().__init__(turn_types=[Turn.SINGLE, Turn.MULTI], backtrack=False)

    def get_description(self) -> ModuleDescriptionHint:
        return [ModuleTag.LLM, ModuleTag.MULTI], "My multi-turn chatbot."

    def get_available_option_values(self) -> ModuleOptionsHint:
        return ["", "proxy=localhost:8080"], False

    def __send(self, session_id: str, message: str, url: str = API_URL) -> str:
        response = requests.post(
            url, headers={"Content-Type": "application/json"},
            json={"thread_id": session_id, "message": message}, timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "")

    def process_input(
        self,
        input_text: str,
        system_message: Optional[str] = None,
        target_options: Optional[str] = None,
        spikee_session_id: Optional[str] = None,
        backtrack: Optional[bool] = False,
    ) -> TargetResponseHint:
        opts = parse_options(target_options)
        url = opts.get("url", API_URL)

        # ---- Session ID ----
        if spikee_session_id is None:
            target_session_id = str(uuid.uuid4())
        else:
            id_map = self._get_id_map(spikee_session_id)
            if id_map is None:
                target_session_id = str(uuid.uuid4())
                self._update_id_map(spikee_session_id, [target_session_id])
            else:
                target_session_id = id_map[-1]

        # ---- Backtrack: replay history in a fresh session ----
        if backtrack and spikee_session_id is not None:
            id_map = self._get_id_map(spikee_session_id)
            if id_map is not None:
                history = self._get_conversation_data(spikee_session_id)
                if history and len(history) >= 2:
                    history = history[:-2]
                    new_id = str(uuid.uuid4())
                    for entry in history:
                        if entry["role"] == "user":
                            self.__send(new_id, entry["content"], url=url)
                    target_session_id = new_id
                    id_map.append(new_id)
                    self._update_id_map(spikee_session_id, id_map)
                    self._update_conversation_data(spikee_session_id, history)

        # ---- Send ----
        try:
            reply = self.__send(target_session_id, input_text, url=url)
        except requests.exceptions.RequestException as e:
            if hasattr(e, "response") and e.response is not None:
                if e.response.status_code == 400:
                    raise GuardrailTrigger(f"Guardrail triggered: {e}")
            print(f"HTTP error: {e}")
            raise

        # ---- Update history ----
        if spikee_session_id is not None:
            self._append_conversation_data(spikee_session_id, role="user", content=input_text)
            self._append_conversation_data(spikee_session_id, role="assistant", content=reply)

        return reply


if __name__ == "__main__":
    target = MyChatbotTarget()
    target.add_managed_dicts({})  # required for standalone use
    sid = "test-session"
    print(target.process_input("Hello, my name is Spikee.", spikee_session_id=sid))
    print(target.process_input("What was my name?", spikee_session_id=sid))
```

---

## 8. API / Transport Patterns

### 8.1 JSON POST (most common)

```python
response = requests.post(url, headers=headers, json={"message": input_text}, timeout=30)
response.raise_for_status()
answer = response.json()["answer"]
```

### 8.2 OpenAI-Compatible Chat Completion

```python
payload = {
    "messages": [
        {"role": "system", "content": system_message or "You are a helpful assistant."},
        {"role": "user",   "content": input_text},
    ],
    "max_tokens": 800, "temperature": 0.7,
}
answer = requests.post(url, headers=headers, json=payload, timeout=30
                       ).json()["choices"][0]["message"]["content"]
```

### 8.3 tRPC Batched Request

URL: `POST /trpc/<procedure>?batch=1`. Body keyed `"0"`, response is a JSON array.

```python
payload = {"0": {"prompt": input_text, "documents": [], "projectId": PROJECT_ID}}
result = requests.post(url, headers=headers, json=payload, timeout=30
                       ).json()[0]["result"]["data"]["fieldYouWant"]
```

### 8.4 Multipart / File Upload

```python
from fpdf import FPDF
pdf = FPDF(); pdf.add_page(); pdf.set_font("Arial", size=12)
pdf.multi_cell(0, 10, input_text)
pdf_bytes = pdf.output(dest="S").encode("latin1")
response = requests.post(url,
    files={"file": ("document.pdf", pdf_bytes, "application/pdf")},
    data={"payload": json.dumps({"question": "Summarise this."})}, timeout=30)
```

### 8.5 Streaming (NDJSON / SSE)

```python
answer = ""
for line in response.text.strip().split("\n"):
    if line:
        chunk = json.loads(line)
        if "content" in chunk:
            answer += chunk["content"]
```

### 8.6 WebSocket (async)

```python
import asyncio, websockets

async def _ws_exchange(self, uri, message, cookies):
    async with websockets.connect(uri, additional_headers={"Cookie": cookies}) as ws:
        await ws.send(json.dumps({"channel": "/meta/handshake", ...}))
        client_id = json.loads(await ws.recv())[0]["clientId"]
        await ws.send(json.dumps({"channel": "/chat/messages",
                                  "data": {"text": message}, "clientId": client_id}))
        async with asyncio.timeout(30):
            async for raw in ws:
                data = json.loads(raw)
                if data[0].get("channel") == "/chat/messages":
                    return data[0]["data"]["text"]

def process_input(self, input_text, system_message=None, target_options=None):
    self.get_auth()
    return asyncio.run(self._ws_exchange(self._ws_uri, input_text, self._cookie_str))
```

### 8.7 WebSocket (sync, websocket-client)

```python
import websocket
ws = websocket.create_connection(url, http_proxy_host="127.0.0.1",
     http_proxy_port=8080, sslopt={"cert_reqs": ssl.CERT_NONE})
ws.send(json.dumps(payload))
response = ws.recv()
ws.close()
```

---

## 9. Proxy and TLS

```python
from urllib3.exceptions import InsecureRequestWarning
requests.packages.urllib3.disable_warnings(category=InsecureRequestWarning)

# Standard pattern: opt-in via target_options
opts = parse_options(target_options)
proxy_host = opts.get("proxy")  # e.g. --target-options "proxy=localhost:8080"
proxies = {"http": f"http://{proxy_host}", "https": f"http://{proxy_host}"} if proxy_host else {}
verify  = not bool(proxy_host)

response = requests.post(url, ..., proxies=proxies, verify=verify, timeout=30)
```

---

## 10. Decision Tree

```
What kind of target?
│
├── Guardrail / classifier
│   └── Inherit Target → return bool (True = bypassed, False = blocked)
│
└── LLM / chatbot (returns text)
    │
    ├── Single message, no history → Inherit Target
    │
    └── Conversations with history
        ├── Standard history tracking  → Inherit SimpleMultiTarget
        ├── Complex / custom state     → Inherit MultiTarget
        └── Stateless API, client-side history → SimpleMultiTarget + replay on backtrack
```
