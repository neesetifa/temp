
import asyncio
from typing import Callable

import pytest

from async_lru import alru_cache


class ControlledCalls:
    """Create one externally controlled Future per wrapped invocation."""

    def __init__(self) -> None:
        self.started: asyncio.Queue[
            tuple[str, asyncio.Future[str]]
        ] = asyncio.Queue()

    async def run(self, key: str) -> str:
        gate: asyncio.Future[str] = (
            asyncio.get_running_loop().create_future()
        )
        self.started.put_nowait((key, gate))
        return await gate

    async def next(self, expected_key: str) -> asyncio.Future[str]:
        key, gate = await self.started.get()
        assert key == expected_key
        return gate


class ManualHandle:
    """A deterministic replacement for an asyncio timer handle."""

    def __init__(
        self,
        callback: Callable[..., object],
        args: tuple[object, ...],
    ) -> None:
        self._callback = callback
        self._args = args
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def fire(self) -> None:
        if not self._cancelled:
            self._callback(*self._args)


async def test_cancelled_detached_caller_does_not_remove_replacement() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_call = asyncio.create_task(cached("key"))
    await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    old_call.cancel()

    with pytest.raises(asyncio.CancelledError):
        await old_call

    await asyncio.sleep(0)

    try:
        assert cached.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"


async def test_cancelled_detached_computation_does_not_remove_replacement() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_call = asyncio.create_task(cached("key"))
    old_gate = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    # Cancel the Future awaited by the underlying old computation.
    old_gate.cancel()

    with pytest.raises(asyncio.CancelledError):
        await old_call

    await asyncio.sleep(0)

    try:
        assert cached.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"


async def test_last_detached_waiter_cannot_remove_replacement() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    first_waiter = asyncio.create_task(cached("key"))
    await calls.next("key")

    second_waiter = asyncio.create_task(cached("key"))
    await asyncio.sleep(0)

    # Both waiters share the old generation.
    assert calls.started.empty()

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    first_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first_waiter

    # One old waiter still remains, so this should not affect replacement.
    assert cached.cache_contains("key")

    second_waiter.cancel()
    with pytest.raises(asyncio.CancelledError):
        await second_waiter

    await asyncio.sleep(0)

    try:
        assert cached.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"


async def test_detached_success_does_not_install_ttl_on_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = ControlledCalls()

    @alru_cache(ttl=60)
    async def cached(key: str) -> str:
        return await calls.run(key)

    loop = asyncio.get_running_loop()
    handles: list[ManualHandle] = []

    def fake_call_later(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> ManualHandle:
        assert delay == 60

        handle = ManualHandle(callback, args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", fake_call_later)

    old_call = asyncio.create_task(cached("key"))
    old_gate = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    old_gate.set_result("old")
    assert await old_call == "old"

    await asyncio.sleep(0)

    # Fire any timer registered by the detached old generation.
    stale_handles = list(handles)
    for handle in stale_handles:
        handle.fire()

    try:
        # A timer originating from the old generation must not expire
        # the in-flight replacement.
        assert cached.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"

    await asyncio.sleep(0)

    # The replacement must still install and honor its own TTL.
    replacement_handles = handles[len(stale_handles) :]
    assert len(replacement_handles) == 1

    replacement_handles[0].fire()
    assert not cached.cache_contains("key")


async def test_cache_clear_detachment_is_generation_isolated() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_call = asyncio.create_task(cached("key"))
    old_gate = await calls.next("key")

    cached.cache_clear()

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    old_gate.set_exception(RuntimeError("cleared generation failed"))

    with pytest.raises(RuntimeError, match="cleared generation failed"):
        await old_call

    await asyncio.sleep(0)

    try:
        assert cached.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"


async def test_evicted_inflight_entry_cannot_remove_reinserted_key() -> None:
    calls = ControlledCalls()

    @alru_cache(maxsize=1)
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_x = asyncio.create_task(cached("x"))
    old_x_gate = await calls.next("x")

    # Inserting y evicts the original x entry.
    y_call = asyncio.create_task(cached("y"))
    y_gate = await calls.next("y")

    assert not cached.cache_contains("x")
    assert cached.cache_contains("y")

    # Reinserting x creates a new generation and evicts y.
    new_x = asyncio.create_task(cached("x"))
    new_x_gate = await calls.next("x")

    assert cached.cache_contains("x")
    assert not cached.cache_contains("y")

    old_x_gate.set_exception(
        RuntimeError("evicted generation failed")
    )

    with pytest.raises(RuntimeError, match="evicted generation failed"):
        await old_x

    await asyncio.sleep(0)

    try:
        assert cached.cache_contains("x")
    finally:
        y_gate.set_result("y")
        new_x_gate.set_result("new x")

        assert await y_call == "y"
        assert await new_x == "new x"


async def test_active_replacement_failure_removes_its_own_entry() -> None:
    """Positive control: identity protection must not retain active failures."""

    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_call = asyncio.create_task(cached("key"))
    old_gate = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    old_gate.set_result("old")
    assert await old_call == "old"

    await asyncio.sleep(0)
    assert cached.cache_contains("key")

    replacement_gate.set_exception(
        ValueError("replacement failed")
    )

    with pytest.raises(ValueError, match="replacement failed"):
        await replacement

    await asyncio.sleep(0)

    assert not cached.cache_contains("key")


async def test_async_method_replacement_is_generation_isolated() -> None:
    calls = ControlledCalls()

    class Service:
        @alru_cache()
        async def load(self, key: str) -> str:
            return await calls.run(key)

    service = Service()

    old_call = asyncio.create_task(service.load("key"))
    old_gate = await calls.next("key")

    assert service.load.cache_invalidate("key")

    replacement = asyncio.create_task(service.load("key"))
    replacement_gate = await calls.next("key")

    old_gate.set_exception(RuntimeError("old method call failed"))

    with pytest.raises(RuntimeError, match="old method call failed"):
        await old_call

    await asyncio.sleep(0)

    try:
        assert service.load.cache_contains("key")
    finally:
        replacement_gate.set_result("replacement")
        assert await replacement == "replacement"
async def test_multiple_detached_generations_cannot_affect_current_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = ControlledCalls()

    @alru_cache(ttl=60)
    async def cached(key: str) -> str:
        return await calls.run(key)

    loop = asyncio.get_running_loop()
    handles: list[ManualHandle] = []

    def fake_call_later(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> ManualHandle:
        handle = ManualHandle(callback, args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", fake_call_later)

    generation_a = asyncio.create_task(cached("key"))
    gate_a = await calls.next("key")

    assert cached.cache_invalidate("key")

    generation_b = asyncio.create_task(cached("key"))
    gate_b = await calls.next("key")

    assert cached.cache_invalidate("key")

    generation_c = asyncio.create_task(cached("key"))
    gate_c = await calls.next("key")

    gate_b.set_exception(RuntimeError("generation b failed"))
    with pytest.raises(RuntimeError, match="generation b failed"):
        await generation_b

    gate_a.set_result("generation a")
    assert await generation_a == "generation a"

    await asyncio.sleep(0)

    # Fire every timer created by either detached generation.
    for handle in list(handles):
        handle.fire()

    try:
        assert cached.cache_contains("key")
    finally:
        gate_c.set_result("generation c")
        assert await generation_c == "generation c"
async def test_detached_tasks_remain_managed_by_cache_close() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_call = asyncio.create_task(cached("key"))
    await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    await calls.next("key")

    await cached.cache_close(wait=False)

    with pytest.raises(asyncio.CancelledError):
        await old_call

    with pytest.raises(asyncio.CancelledError):
        await replacement
async def test_detached_completion_does_not_cancel_replacement_ttl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = ControlledCalls()

    @alru_cache(ttl=60)
    async def cached(key: str) -> str:
        return await calls.run(key)

    loop = asyncio.get_running_loop()
    handles: list[ManualHandle] = []

    def fake_call_later(
        delay: float,
        callback: Callable[..., object],
        *args: object,
    ) -> ManualHandle:
        handle = ManualHandle(callback, args)
        handles.append(handle)
        return handle

    monkeypatch.setattr(loop, "call_later", fake_call_later)

    old_call = asyncio.create_task(cached("key"))
    old_gate = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement = asyncio.create_task(cached("key"))
    replacement_gate = await calls.next("key")

    replacement_gate.set_result("replacement")
    assert await replacement == "replacement"

    await asyncio.sleep(0)
    assert len(handles) == 1
    replacement_handle = handles[0]

    old_gate.set_result("old")
    assert await old_call == "old"

    await asyncio.sleep(0)

    assert not replacement_handle._cancelled
    assert cached.cache_contains("key")

    replacement_handle.fire()
    assert not cached.cache_contains("key")
