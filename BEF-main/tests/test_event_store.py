from pathlib import Path

from server.event_store import EventStore


def test_event_store_append_and_history(tmp_path: Path) -> None:
    store = EventStore(tmp_path)
    store.append("run1", "{\"seq\":1}")
    store.append("run1", "{\"seq\":2}")
    assert store.history("run1") == ["{\"seq\":1}", "{\"seq\":2}"]
    assert store.history("run1", last_seq=1) == ["{\"seq\":2}"]
