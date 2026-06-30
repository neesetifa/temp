```python
from __future__ import annotations

import copy
import pickle
from typing import Any

import pytest

import h11


def restored_connection(conn: h11.Connection) -> h11.Connection:
    snapshot = conn.snapshot()

    # The snapshot is expected to be suitable for storage or transfer.
    # Running every restore through pickle catches implementations that
    # accidentally return live internal objects.
    snapshot = pickle.loads(pickle.dumps(copy.deepcopy(snapshot)))

    return h11.Connection.from_snapshot(snapshot)


def assert_portable_snapshot(value: Any) -> None:
    primitive_types = (str, bytes, int, bool, type(None))

    if isinstance(value, primitive_types):
        return

    if isinstance(value, (list, tuple)):
        for item in value:
            assert_portable_snapshot(item)
        return

    if isinstance(value, dict):
        for key, item in value.items():
            assert_portable_snapshot(key)
            assert_portable_snapshot(item)
        return

    raise AssertionError(
        f"snapshot contains non-portable object {value!r} "
        f"of type {type(value)!r}"
    )


def next_request(conn: h11.Connection) -> h11.Request:
    event = conn.next_event()
    assert isinstance(event, h11.Request)
    return event


def next_data(conn: h11.Connection) -> h11.Data:
    event = conn.next_event()
    assert isinstance(event, h11.Data)
    return event


def next_end(conn: h11.Connection) -> h11.EndOfMessage:
    event = conn.next_event()
    assert isinstance(event, h11.EndOfMessage)
    return event


def assert_need_data(conn: h11.Connection) -> None:
    assert conn.next_event() is h11.NEED_DATA


# ---------------------------------------------------------------------------
# PUBLIC TEST
# ---------------------------------------------------------------------------


def test_snapshot_preserves_incomplete_request_headers() -> None:
    conn = h11.Connection(
        h11.SERVER,
        max_incomplete_event_size=1024,
    )

    conn.receive_data(b"GET / HTTP/1.1\r\nHost: exam")

    restored = restored_connection(conn)

    for candidate in (conn, restored):
        candidate.receive_data(b"ple.test\r\n\r\n")

        request = next_request(candidate)
        assert request.method == b"GET"
        assert request.target == b"/"
        assert dict(request.headers)[b"host"] == b"example.test"

        next_end(candidate)
        assert_need_data(candidate)


# ---------------------------------------------------------------------------
# HIDDEN TESTS
# ---------------------------------------------------------------------------


def test_snapshot_is_portable_and_restored_connection_is_independent() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(b"GET /")

    snapshot = conn.snapshot()
    assert_portable_snapshot(pickle.loads(pickle.dumps(snapshot)))

    restored = h11.Connection.from_snapshot(
        pickle.loads(pickle.dumps(copy.deepcopy(snapshot)))
    )

    conn.receive_data(b"left HTTP/1.1\r\nHost: example.test\r\n\r\n")
    restored.receive_data(
        b"right HTTP/1.1\r\nHost: example.test\r\n\r\n"
    )

    original_request = next_request(conn)
    restored_request = next_request(restored)

    assert original_request.target == b"/left"
    assert restored_request.target == b"/right"

    next_end(conn)
    next_end(restored)


def test_snapshot_preserves_fixed_length_body_reader_state() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"POST /upload HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"Content-Length: 5\r\n"
        b"\r\n"
        b"ab"
    )

    request = next_request(conn)
    assert request.method == b"POST"
    assert request.target == b"/upload"

    first_piece = next_data(conn)
    assert bytes(first_piece.data) == b"ab"

    assert_need_data(conn)

    restored = restored_connection(conn)

    conn.receive_data(b"XYZ")
    restored.receive_data(b"cde")

    original_remaining = next_data(conn)
    restored_remaining = next_data(restored)

    assert bytes(original_remaining.data) == b"XYZ"
    assert bytes(restored_remaining.data) == b"cde"

    next_end(conn)
    next_end(restored)


def test_snapshot_preserves_chunked_body_reader_state() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"POST /chunked HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"Transfer-Encoding: chunked\r\n"
        b"\r\n"
        b"5\r\nhe"
    )

    request = next_request(conn)
    assert request.target == b"/chunked"

    first_piece = next_data(conn)
    assert bytes(first_piece.data) == b"he"
    assert first_piece.chunk_start is True
    assert first_piece.chunk_end is False

    assert_need_data(conn)

    restored = restored_connection(conn)

    restored.receive_data(b"llo\r\n0\r\nX-Trailer: done\r\n\r\n")

    second_piece = next_data(restored)
    assert bytes(second_piece.data) == b"llo"
    assert second_piece.chunk_start is False
    assert second_piece.chunk_end is True

    end = next_end(restored)
    assert dict(end.headers)[b"x-trailer"] == b"done"


def test_snapshot_preserves_body_writer_state() -> None:
    conn = h11.Connection(h11.CLIENT)

    request_bytes = conn.send(
        h11.Request(
            method=b"POST",
            target=b"/upload",
            headers=[
                (b"Host", b"example.test"),
                (b"Content-Length", b"5"),
            ],
        )
    )
    assert request_bytes.startswith(b"POST /upload HTTP/1.1")

    assert conn.send(h11.Data(data=b"ab")) == b"ab"

    restored = restored_connection(conn)

    assert restored.send(h11.Data(data=b"cde")) == b"cde"
    assert restored.send(h11.EndOfMessage()) == b""

    with pytest.raises(h11.LocalProtocolError):
        restored.send(h11.Data(data=b"!"))


def test_snapshot_preserves_100_continue_state() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"POST /submit HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"Content-Length: 4\r\n"
        b"Expect: 100-continue\r\n"
        b"\r\n"
    )

    request = next_request(conn)
    assert request.target == b"/submit"
    assert conn.they_are_waiting_for_100_continue is True

    restored = restored_connection(conn)

    assert restored.they_are_waiting_for_100_continue is True
    assert conn.they_are_waiting_for_100_continue is True

    continue_bytes = restored.send(
        h11.InformationalResponse(
            status_code=100,
            headers=[],
        )
    )
    assert continue_bytes.startswith(b"HTTP/1.1 100")

    assert restored.they_are_waiting_for_100_continue is False
    assert conn.they_are_waiting_for_100_continue is True

    restored.receive_data(b"data")

    body = next_data(restored)
    assert bytes(body.data) == b"data"

    next_end(restored)


def test_snapshot_preserves_pipelined_trailing_data_and_paused_state() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"GET /one HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"\r\n"
        b"GET /two HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"\r\n"
    )

    first = next_request(conn)
    assert first.target == b"/one"

    next_end(conn)

    assert conn.next_event() is h11.PAUSED

    restored = restored_connection(conn)

    response_bytes = restored.send(
        h11.Response(
            status_code=200,
            headers=[
                (b"Content-Length", b"0"),
            ],
        )
    )
    assert response_bytes.startswith(b"HTTP/1.1 200")

    assert restored.send(h11.EndOfMessage()) == b""

    restored.start_next_cycle()

    second = next_request(restored)
    assert second.target == b"/two"

    next_end(restored)
    assert_need_data(restored)


def test_snapshot_preserves_eof_state_for_incomplete_input() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(b"GET / HTTP/1.1\r\nHost: example")
    conn.receive_data(b"")

    restored = restored_connection(conn)

    for candidate in (restored, conn):
        with pytest.raises(
            h11.RemoteProtocolError,
            match="peer unexpectedly closed connection",
        ):
            candidate.next_event()


def test_snapshot_preserves_max_incomplete_event_size() -> None:
    conn = h11.Connection(
        h11.SERVER,
        max_incomplete_event_size=20,
    )

    conn.receive_data(b"GET /")

    restored = restored_connection(conn)

    restored.receive_data(b"x" * 30)

    with pytest.raises(
        h11.RemoteProtocolError,
        match="Receive buffer too long",
    ):
        restored.next_event()


def test_snapshot_does_not_advance_or_consume_pending_event() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"GET /still-pending HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"\r\n"
    )

    restored = restored_connection(conn)

    original_request = next_request(conn)
    restored_request = next_request(restored)

    assert original_request.target == b"/still-pending"
    assert restored_request.target == b"/still-pending"

    next_end(conn)
    next_end(restored)

    assert_need_data(conn)
    assert_need_data(restored)


def test_snapshot_round_trip_after_start_next_cycle() -> None:
    conn = h11.Connection(h11.SERVER)

    conn.receive_data(
        b"GET /one HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"\r\n"
        b"GET /two HTTP/1.1\r\n"
        b"Host: example.test\r\n"
        b"\r\n"
    )

    first = next_request(conn)
    assert first.target == b"/one"
    next_end(conn)
    assert conn.next_event() is h11.PAUSED

    conn.send(
        h11.Response(
            status_code=200,
            headers=[
                (b"Content-Length", b"0"),
            ],
        )
    )
    conn.send(h11.EndOfMessage())
    conn.start_next_cycle()

    restored = restored_connection(conn)

    second = next_request(restored)
    assert second.target == b"/two"

    next_end(restored)
    assert_need_data(restored)
```
