# Isolate replacement cache entries and retain ownership of detached async tasks

`async-lru` allows an in-flight cached call to be removed from the cache through explicit invalidation, cache clearing, or normal eviction. Removing an entry from the cache lookup does not necessarily stop the underlying asynchronous computation.

A later call using the same arguments may therefore create a replacement cache entry while the earlier computation is still running. The earlier entry is considered **detached**: it is no longer available for cache lookup, but its task may still have active callers and remains part of the cache's lifecycle until it reaches a terminal state.

Update the cache lifecycle so that detached calls cannot corrupt replacement entries and do not become unmanaged.

## Replacement entry isolation

Once an in-flight entry is no longer the active cached entry for its key, later activity associated with that entry must not remove, expire, or otherwise modify a newer entry created for the same key.

The following behavior is required:

* Invalidating an in-flight entry must allow a subsequent call with the same arguments to create and cache a new computation.
* If a detached computation later raises an exception, its cleanup must not remove the replacement entry.
* If a detached computation is later cancelled, its cleanup must not remove the replacement entry.
* Successful completion of a detached computation must not install, reset, cancel, or otherwise alter expiration state belonging to the replacement entry.
* Cleanup triggered by cancellation of callers waiting on a detached computation must not affect the replacement entry.
* The same isolation guarantee must hold when an entry becomes detached through `cache_clear()` or normal cache eviction.
* Multiple detached generations for the same key must remain isolated from the currently active generation and from one another.
* The original detached callers must still observe the result, exception, or cancellation of their own computation.
* Calls sharing the currently active entry must continue to share one underlying task.
* A currently active entry must still be removed according to the existing behavior when its own computation fails or is cancelled.
* A replacement entry must still expire normally according to its own TTL.
* Existing cache hit, miss, invalidation, clearing, eviction, and TTL behavior must remain compatible for entries that have not been replaced.

## Detached task lifecycle management

Removing an in-flight entry from the cache lookup must not cause its underlying task to become unmanaged.

A task created by the cache remains owned by that cache until it completes, fails, or is cancelled, even if the corresponding entry has already been invalidated, cleared, evicted, or replaced.

The following behavior is required:

* Detaching an entry must not by itself cancel its underlying task unless existing behavior already requires cancellation in that operation.
* `cache_close(wait=False)` must cancel and await all unfinished tasks owned by the cache, including tasks belonging to detached entries.
* `cache_close(wait=True)` must wait for all unfinished cache-owned tasks, including detached tasks, without cancelling them.
* Active and detached tasks must not be registered more than once when they are shared by multiple callers or pass through multiple cache lifecycle operations.
* A task must stop being tracked after it reaches a terminal state.
* Tracking detached tasks must not unnecessarily retain completed tasks, cache entries, call arguments, results, exceptions, or bound instances.
* Repeated calls to `cache_close()` must remain safe and idempotent.
* Closing the cache must not allow callbacks from detached tasks to modify replacement entries.
* If the project already exposes task-count or task-reporting behavior, that behavior must account for all unfinished cache-owned tasks, including detached tasks. No new public task-reporting API is required solely for this change.

## Compatibility requirements

* The fix must support both cached coroutine functions and cached async methods.
* Existing behavior for callers currently awaiting a shared task must remain compatible.
* Existing cache statistics must remain accurate.
* Existing public APIs and return types should remain unchanged unless a public API change is necessary for correctness.
* The implementation must not rely on timing-dependent sleeps or nondeterministic scheduling behavior.

## Example scenario

Consider a cached coroutine called repeatedly with the same arguments:

1. The first call starts and remains in flight.
2. Its cache entry is invalidated.
3. A second call with the same arguments starts and becomes the active cached call.
4. The first call later fails.

After the first call fails, another call with the same arguments must continue to share or return the second call's cached computation. Cleanup from the first call must not remove the second entry.

The first task must also remain owned by the cache after invalidation. If the cache is closed before that task finishes, `cache_close()` must manage it according to the requested `wait` mode rather than leaving it orphaned.

The implementation should preserve existing asynchronous and cancellation semantics while ensuring that:

1. cache-entry lifecycle actions apply only to the entry they belong to; and
2. every unfinished task created by the cache remains under cache lifecycle management until termination.



```python
import asyncio
import gc
import weakref
from typing import Any, cast

import pytest

from async_lru import alru_cache


class ControlledCalls:
    """Create one externally controlled Future per underlying invocation."""

    def __init__(self) -> None:
        self.started: asyncio.Queue[
            tuple[
                str,
                asyncio.Future[str],
                asyncio.Task[str],
            ]
        ] = asyncio.Queue()

    async def run(self, key: str) -> str:
        loop = asyncio.get_running_loop()
        gate: asyncio.Future[str] = loop.create_future()

        current = asyncio.current_task()
        assert current is not None

        self.started.put_nowait(
            (
                key,
                gate,
                cast(asyncio.Task[str], current),
            )
        )
        return await gate

    async def next(
        self,
        expected_key: str,
    ) -> tuple[asyncio.Future[str], asyncio.Task[str]]:
        key, gate, task = await self.started.get()
        assert key == expected_key
        return gate, task


async def drain_callbacks() -> None:
    """Allow task completion and cache callbacks to run."""

    await asyncio.sleep(0)
    await asyncio.sleep(0)


async def cancel_and_drain(
    *tasks: asyncio.Task[Any],
) -> None:
    for task in tasks:
        if not task.done():
            task.cancel()

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)

    await drain_callbacks()


async def test_cache_close_cancels_detached_and_active_tasks() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_waiter = asyncio.create_task(cached("key"))
    _, old_task = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement_waiter = asyncio.create_task(cached("key"))
    _, replacement_task = await calls.next("key")

    try:
        await cached.cache_close(wait=False)
        await drain_callbacks()

        # cache_close() owns both tasks, even though only the replacement
        # remains reachable through the cache lookup.
        assert old_task.cancelled()
        assert replacement_task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await old_waiter

        with pytest.raises(asyncio.CancelledError):
            await replacement_waiter

        assert cached.cache_parameters()["tasks"] == 0
        assert cached.cache_parameters()["closed"] is True

        # Closing an already closed cache remains safe.
        await cached.cache_close(wait=False)
        await cached.cache_close(wait=True)
    finally:
        await cancel_and_drain(old_waiter, replacement_waiter)


async def test_cache_close_waits_for_detached_tasks_without_cancelling() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    old_waiter = asyncio.create_task(cached("key"))
    old_gate, old_task = await calls.next("key")

    assert cached.cache_invalidate("key")

    replacement_waiter = asyncio.create_task(cached("key"))
    replacement_gate, replacement_task = await calls.next("key")

    close_task = asyncio.create_task(cached.cache_close(wait=True))

    try:
        await drain_callbacks()

        assert not close_task.done()
        assert not old_task.cancelled()
        assert not replacement_task.cancelled()
        assert not old_gate.cancelled()
        assert not replacement_gate.cancelled()

        # Completing only the active replacement must not allow close()
        # to return while the detached generation is still running.
        replacement_gate.set_result("replacement")
        assert await replacement_waiter == "replacement"

        await drain_callbacks()

        assert not close_task.done()
        assert not old_task.done()
        assert cached.cache_parameters()["tasks"] == 1

        old_gate.set_result("old")
        assert await old_waiter == "old"

        await close_task

        assert cached.cache_parameters()["tasks"] == 0
        assert cached.cache_parameters()["closed"] is True
    finally:
        await cancel_and_drain(
            old_waiter,
            replacement_waiter,
            close_task,
        )


async def test_task_reporting_includes_multiple_detached_generations() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    generation_a = asyncio.create_task(cached("key"))
    gate_a, _ = await calls.next("key")

    assert cached.cache_invalidate("key")

    generation_b = asyncio.create_task(cached("key"))
    gate_b, _ = await calls.next("key")

    # B becomes detached through cache clearing rather than invalidation.
    cached.cache_clear()

    generation_c = asyncio.create_task(cached("key"))
    _, _ = await calls.next("key")

    try:
        # A and B are detached; C is active.
        assert cached.cache_parameters()["tasks"] == 3

        gate_a.set_result("generation a")
        assert await generation_a == "generation a"

        await drain_callbacks()
        assert cached.cache_parameters()["tasks"] == 2

        gate_b.set_exception(RuntimeError("generation b failed"))

        with pytest.raises(
            RuntimeError,
            match="generation b failed",
        ):
            await generation_b

        await drain_callbacks()

        # The detached B failure must not remove or untrack C.
        assert cached.cache_contains("key")
        assert cached.cache_parameters()["tasks"] == 1

        generation_c.cancel()

        with pytest.raises(asyncio.CancelledError):
            await generation_c

        await drain_callbacks()
        assert cached.cache_parameters()["tasks"] == 0
    finally:
        await cancel_and_drain(
            generation_a,
            generation_b,
            generation_c,
        )


async def test_evicted_task_remains_owned_by_cache_close() -> None:
    calls = ControlledCalls()

    @alru_cache(maxsize=1)
    async def cached(key: str) -> str:
        return await calls.run(key)

    evicted_waiter = asyncio.create_task(cached("x"))
    _, evicted_task = await calls.next("x")

    active_waiter = asyncio.create_task(cached("y"))
    _, active_task = await calls.next("y")

    try:
        assert not cached.cache_contains("x")
        assert cached.cache_contains("y")

        # The evicted X task and active Y task are both still unfinished.
        assert cached.cache_parameters()["tasks"] == 2

        await cached.cache_close(wait=False)
        await drain_callbacks()

        assert evicted_task.cancelled()
        assert active_task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await evicted_waiter

        with pytest.raises(asyncio.CancelledError):
            await active_waiter

        assert cached.cache_parameters()["tasks"] == 0
    finally:
        await cancel_and_drain(evicted_waiter, active_waiter)


async def test_cache_clear_does_not_orphan_inflight_tasks() -> None:
    calls = ControlledCalls()

    @alru_cache(maxsize=None)
    async def cached(key: str) -> str:
        return await calls.run(key)

    first_waiter = asyncio.create_task(cached("first"))
    _, first_task = await calls.next("first")

    second_waiter = asyncio.create_task(cached("second"))
    _, second_task = await calls.next("second")

    cached.cache_clear()

    try:
        assert not cached.cache_contains("first")
        assert not cached.cache_contains("second")
        assert cached.cache_parameters()["tasks"] == 2

        await cached.cache_close(wait=False)
        await drain_callbacks()

        assert first_task.cancelled()
        assert second_task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await first_waiter

        with pytest.raises(asyncio.CancelledError):
            await second_waiter

        assert cached.cache_parameters()["tasks"] == 0
    finally:
        await cancel_and_drain(first_waiter, second_waiter)


async def test_shared_callers_do_not_duplicate_task_tracking() -> None:
    calls = ControlledCalls()

    @alru_cache()
    async def cached(key: str) -> str:
        return await calls.run(key)

    first_waiter = asyncio.create_task(cached("key"))
    _, original_task = await calls.next("key")

    second_waiter = asyncio.create_task(cached("key"))
    await drain_callbacks()

    # The second caller shares the existing underlying task.
    assert calls.started.empty()
    assert cached.cache_parameters()["tasks"] == 1

    assert cached.cache_invalidate("key")

    # Detaching the task must not duplicate or remove its ownership record.
    assert cached.cache_parameters()["tasks"] == 1

    replacement_waiter = asyncio.create_task(cached("key"))
    _, replacement_task = await calls.next("key")

    try:
        assert original_task is not replacement_task
        assert cached.cache_parameters()["tasks"] == 2

        # Detach the replacement as well. There are still exactly two
        # underlying tasks, despite three logical callers.
        assert cached.cache_invalidate("key")
        assert cached.cache_parameters()["tasks"] == 2

        await cached.cache_close(wait=False)
        await drain_callbacks()

        assert original_task.cancelled()
        assert replacement_task.cancelled()
        assert cached.cache_parameters()["tasks"] == 0

        for waiter in (
            first_waiter,
            second_waiter,
            replacement_waiter,
        ):
            with pytest.raises(asyncio.CancelledError):
                await waiter
    finally:
        await cancel_and_drain(
            first_waiter,
            second_waiter,
            replacement_waiter,
        )


async def test_completed_detached_tasks_are_not_retained() -> None:
    class Payload:
        pass

    started = asyncio.Event()
    release = asyncio.Event()

    @alru_cache()
    async def cached(value: Payload) -> Payload:
        started.set()
        await release.wait()
        return value

    payload = Payload()
    payload_ref = weakref.ref(payload)

    waiter = asyncio.create_task(cached(payload))
    await started.wait()

    assert cached.cache_invalidate(payload)
    assert cached.cache_parameters()["tasks"] == 1

    release.set()
    assert await waiter is payload

    await drain_callbacks()

    assert cached.cache_parameters()["tasks"] == 0

    # The completed Task's result refers to payload. If the cache keeps the
    # terminal Task in its ownership collection, payload remains alive.
    del waiter
    del payload

    for _ in range(3):
        gc.collect()
        await asyncio.sleep(0)

    assert payload_ref() is None

    await cached.cache_close(wait=True)
    await cached.cache_close(wait=False)

    assert cached.cache_parameters()["tasks"] == 0
    assert cached.cache_parameters()["closed"] is True


async def test_async_method_close_manages_detached_generations() -> None:
    calls = ControlledCalls()

    class Service:
        @alru_cache()
        async def load(self, key: str) -> str:
            return await calls.run(key)

    service = Service()

    old_waiter = asyncio.create_task(service.load("key"))
    _, old_task = await calls.next("key")

    assert service.load.cache_invalidate("key")

    replacement_waiter = asyncio.create_task(service.load("key"))
    _, replacement_task = await calls.next("key")

    try:
        assert service.load.cache_parameters()["tasks"] == 2

        await service.load.cache_close(wait=False)
        await drain_callbacks()

        assert old_task.cancelled()
        assert replacement_task.cancelled()

        with pytest.raises(asyncio.CancelledError):
            await old_waiter

        with pytest.raises(asyncio.CancelledError):
            await replacement_waiter

        assert service.load.cache_parameters()["tasks"] == 0
    finally:
        await cancel_and_drain(old_waiter, replacement_waiter)
```
