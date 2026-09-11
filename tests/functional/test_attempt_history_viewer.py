"""Candidate evidence stays nested under a single result in the viewer."""

from copy import deepcopy
from html.parser import HTMLParser

import pytest
from werkzeug.datastructures import MultiDict

from spikee.templates.standardised_conversation import StandardisedConversation
from spikee.utilities.files import read_jsonl_file, write_jsonl_file
from spikee.viewer.app import create_app
from spikee.viewer.blueprints import _cache
from spikee.viewer.blueprints import results as viewer_results
from spikee.viewer.blueprints._forms import TestForm as RunForm


class ControlMarkup(HTMLParser):
    """Collect real elements so escaped payload text cannot satisfy assertions."""

    def __init__(self, html):
        super().__init__()
        self.elements = []
        self.feed(html)

    def handle_starttag(self, tag, attrs):
        self.elements.append((tag, dict(attrs)))

    def controls(self):
        return [
            attrs
            for tag, attrs in self.elements
            if tag == "details" and "conversation-card" in attrs.get("class", "").split()
        ]


@pytest.fixture
def history_viewer(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "results").mkdir()
    (tmp_path / "datasets").mkdir()
    history = [
        {
            "input": "candidate prompt <script>alert('input')</script>",
            "response": "candidate response <img src=x onerror=alert('response')>",
            "success": False,
        },
        {
            "input": ["structured input", {"text": "another part"}],
            "response": {"text": "unjudged response"},
            "success": None,
        },
        {
            "input": "error candidate",
            "response": "response before judge error",
            "error": "judge failed <script>alert('error')</script>",
        },
        {"input": "successful candidate", "response": "done", "success": True},
    ]
    row = {
        "id": "42-attack",
        "long_id": "dataset-entry-best_of_n",
        "input": "successful candidate",
        "response": "done",
        "success": True,
        "attempts": 4,
        "attack_name": "best_of_n",
        "judge_name": "canary",
        "judge_args": "done",
        "attempt_history": history,
    }
    path = tmp_path / "results" / "results_history.jsonl"
    write_jsonl_file(path, [row])
    monkeypatch.setattr(_cache, "warm_cache", lambda: None)
    monkeypatch.setattr(viewer_results, "loaded_files", {"history": path})
    app = create_app(db_path=str(tmp_path / "jobs.sqlite"))
    app.config["TESTING"] = True
    return app.test_client(), path, row


def test_candidate_history_detail_is_escaped_and_summary_stays_compact(history_viewer):
    client, _path, _row = history_viewer
    response = client.get("/results/entries?result_file=history")
    assert response.status_code == 200
    assert b"&middot; 4 candidates" in response.data
    assert b"View history" not in response.data
    assert b'id="attempt-history"' not in response.data
    assert b"candidate prompt &lt;script&gt;" in response.data
    details = [
        attrs for tag, attrs in ControlMarkup(response.get_data(as_text=True)).elements
        if tag == "details"
    ]
    assert len(details) == 5
    assert all("open" not in attrs for attrs in details)

    response = client.get("/results/entry/history-42-attack?result_file=history")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert 'id="attempt-history"' in html
    for marker in (
        "Candidate 1",
        "Candidate 4",
        "FAILED",
        "SUCCESS",
        "UNJUDGED",
        "ERROR",
        "structured input",
        "unjudged response",
        "response before judge error",
        "candidate prompt &lt;script&gt;",
        "candidate response &lt;img",
        "judge failed &lt;script&gt;",
    ):
        assert marker in html
    assert "<script>alert(" not in html
    assert "<img src=x" not in html
    assert html.count("Toggle Success") == 1
    assert html.count(">Re-judge<") == 1


def test_rejudge_and_toggle_change_only_the_result(history_viewer, monkeypatch):
    client, path, row = history_viewer
    original_history = deepcopy(row["attempt_history"])
    judged = []

    def judge(entry, response):
        judged.append((entry["id"], response))
        return False

    monkeypatch.setattr(viewer_results, "call_judge", judge)
    route = "/results/entry/history-42-attack"
    response = client.post(f"{route}/rejudge?result_file=history")
    assert response.status_code == 302
    assert judged == [("42-attack", "done")]
    updated = read_jsonl_file(path)
    assert len(updated) == 1
    assert updated[0]["success"] is False
    assert updated[0]["attempts"] == 4
    assert updated[0]["attempt_history"] == original_history
    response = client.post(f"{route}/toggle?result_file=history")
    assert response.status_code == 302
    updated = read_jsonl_file(path)
    assert updated[0]["success"] is True
    assert updated[0]["attempt_history"] == original_history


@pytest.mark.parametrize("format", ["representative", "conversation", "expanded"])
def test_existing_result_formats_still_render(history_viewer, format):
    client, path, row = history_viewer
    del row["attempt_history"]
    if format == "conversation":
        conversation = StandardisedConversation("conversation objective")
        conversation.add_message(0, {"response": "preserved conversation branch"})
        row["conversation"] = str(conversation)
    elif format == "expanded":
        row.update(
            id="42-attack-1",
            attack_attempt=1,
            attack_parent_id=42,
            attack_result_format="attempt",
        )
    write_jsonl_file(path, [row])
    response = client.get(f"/results/entry/history-{row['id']}?result_file=history")
    assert response.status_code == 200
    assert b'id="attempt-history"' not in response.data
    if format == "conversation":
        assert b"Conversation" in response.data
        assert b"preserved conversation branch" in response.data
    elif format == "expanded":
        assert b"Attack Attempt" in response.data


def test_test_form_ignores_saved_retention_flag(history_viewer):
    client, _path, _row = history_viewer
    form = MultiDict(
        {
            "target": "mock",
            "datasets": "example.jsonl",
            "attack": "best_of_n",
            "attack_return_all_attempts": "on",
        }
    )
    args = RunForm.from_form(form).to_cli_args()
    assert args[:3] == ["test", "--target", "mock"]
    assert "--attack" in args
    assert "--attack-return-all-attempts" not in args
    with client.session_transaction() as session:
        session["test_settings"] = dict(form)
    response = client.get("/test/run")
    assert response.status_code == 200
    assert b"attack_return_all_attempts" not in response.data


@pytest.mark.parametrize(
    "format", ["conversation", "mapping", "input_list", "response_list"]
)
@pytest.mark.parametrize("count", [0, 8, 9])
def test_conversation_defaults_and_counts(history_viewer, format, count):
    client, path, row = history_viewer
    payload = "message <script>alert('trace')</script>" + "x" * 2000 + "end-of-message"
    if format == "conversation":
        conversation = StandardisedConversation("root objective")
        for index in range(count):
            conversation.add_message(index, {"response": payload})
        row["conversation"] = str(conversation)
        heading = "Conversation"
    elif format == "mapping":
        row["input"] = {"conversation": [{"role": "user", "content": payload}] * count}
        heading = "Input conversation"
    elif format == "input_list":
        row["input"] = [payload] * count
        heading = "Input conversation"
    else:
        row["response"] = [{"assistant": payload}] * count
        heading = "Response conversation"
    write_jsonl_file(path, [row])

    for route, expanded in [
        ("/results/entries", count <= 8),
        ("/results/entry/history-42-attack", True),
    ]:
        response = client.get(f"{route}?result_file=history")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert heading in html
        assert f"{count} messages" in html
        markup = ControlMarkup(html)
        [panel] = markup.controls()
        assert ("open" in panel) == expanded
        assert "data-bs-toggle" not in panel
        assert "Show trace" not in html
        assert "Hide conversation" not in html
        if count:
            assert "message &lt;script&gt;" in html
            assert html.count("end-of-message") == count
        assert "<script>alert('trace')" not in html


def test_malformed_trace_remains_escaped(history_viewer):
    client, path, row = history_viewer
    row["conversation"] = "invalid <script>alert('trace')</script>"
    write_jsonl_file(path, [row])
    for route in ("/results/entries", "/results/entry/history-42-attack"):
        response = client.get(f"{route}?result_file=history")
        assert response.status_code == 200
        html = response.get_data(as_text=True)
        assert "Conversation" in html
        assert "0 messages" in html
        assert "invalid &lt;script&gt;" in html
        assert "open" in ControlMarkup(html).controls()[0]


def test_conversation_cards_are_independent_across_files(history_viewer, monkeypatch):
    client, path, row = history_viewer
    row.update(id="same.id:[]", input=["input"] * 9, response=["response"] * 9)
    other = path.with_name("results_other.jsonl")
    write_jsonl_file(path, [row])
    write_jsonl_file(other, [row])
    monkeypatch.setattr(
        viewer_results, "loaded_files", {"history": path, "other": other}
    )
    response = client.get("/results/entries")
    assert response.status_code == 200
    controls = ControlMarkup(response.get_data(as_text=True)).controls()
    assert len(controls) == 4
    # Native details have independent state without shared IDs or radio-group names.
    assert all("name" not in panel and "open" not in panel for panel in controls)
    assert all("data-bs-target" not in panel for panel in controls)


@pytest.mark.parametrize("count", [0, 1, 4])
def test_history_bulk_controls_and_initial_state(history_viewer, count):
    client, path, row = history_viewer
    row["attempt_history"] = row["attempt_history"][:count]
    write_jsonl_file(path, [row])
    response = client.get("/results/entry/history-42-attack?result_file=history")
    assert response.status_code == 200
    html = response.get_data(as_text=True)
    assert ("Expand all" in html) == (count > 1)
    assert ("Collapse all" in html) == (count > 1)
    details = [attrs for tag, attrs in ControlMarkup(html).elements if tag == "details"]
    assert len(details) == (count + 1 if count else 0)
    assert all("open" not in attrs for attrs in details)


def test_structured_response_is_not_a_conversation(history_viewer):
    client, path, row = history_viewer
    row["response"] = {"text": "<script>structured</script>"}
    write_jsonl_file(path, [row])
    response = client.get("/results/entry/history-42-attack?result_file=history")
    html = response.get_data(as_text=True)
    assert response.status_code == 200
    assert "structured JSON object" in html
    assert not ControlMarkup(html).controls()
    assert "<script>structured</script>" not in html
