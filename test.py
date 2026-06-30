```python
from __future__ import annotations

import pytest

import pluggy


PROJECT = "rollback"
hookspec = pluggy.HookspecMarker(PROJECT)
hookimpl = pluggy.HookimplMarker(PROJECT)


def assert_pending_unknown_hook(
    pm: pluggy.PluginManager,
    name: str,
) -> None:
    with pytest.raises(
        pluggy.PluginValidationError,
        match=f"unknown hook {name!r}",
    ):
        pm.check_pending()


# ---------------------------------------------------------------------------
# PUBLIC TEST
# ---------------------------------------------------------------------------


def test_failed_hookspec_addition_keeps_pending_hook_unresolved() -> None:
    class Plugin:
        @hookimpl
        def target(self, actual):
            return actual

    class BadSpec:
        @hookspec
        def target(self, expected):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    assert_pending_unknown_hook(pm, "target")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    # The failed spec update must not make the unknown hook look specified.
    assert_pending_unknown_hook(pm, "target")


# ---------------------------------------------------------------------------
# HIDDEN TESTS
# ---------------------------------------------------------------------------


def test_failed_hookspec_addition_can_be_retried_with_compatible_spec() -> None:
    class Plugin:
        @hookimpl
        def target(self, actual):
            return f"seen:{actual}"

    class BadSpec:
        @hookspec
        def target(self, expected):
            pass

    class GoodSpec:
        @hookspec
        def target(self, actual):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    pm.add_hookspecs(GoodSpec)

    assert pm.hook.target(actual="value") == ["seen:value"]


def test_failed_namespace_update_rolls_back_specs_added_earlier() -> None:
    class Plugin:
        @hookimpl
        def zzz_broken(self, extra):
            return extra

    class MixedSpecs:
        @hookspec
        def aaa_new_hook(self, value):
            pass

        @hookspec
        def zzz_broken(self, value):
            pass

    class OnlyNewHookSpec:
        @hookspec
        def aaa_new_hook(self, value):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    assert not hasattr(pm.hook, "aaa_new_hook")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(MixedSpecs)

    # The namespace update failed as a whole. A hookspec which happened
    # to be processed before the failing one must not remain installed.
    assert not hasattr(pm.hook, "aaa_new_hook")

    pm.add_hookspecs(OnlyNewHookSpec)
    assert pm.hook.aaa_new_hook(value=1) == []


def test_failed_namespace_update_rolls_back_historic_state() -> None:
    class Plugin:
        @hookimpl
        def aaa_event(self, value):
            return f"event:{value}"

        @hookimpl
        def zzz_broken(self, extra):
            return extra

    class MixedSpecs:
        @hookspec(historic=True)
        def aaa_event(self, value):
            pass

        @hookspec
        def zzz_broken(self, value):
            pass

    class PlainEventSpec:
        @hookspec
        def aaa_event(self, value):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(MixedSpecs)

    # The failed namespace update must not leave aaa_event as a historic
    # hook or with call-history state attached.
    pm.add_hookspecs(PlainEventSpec)

    assert pm.hook.aaa_event(value="ok") == ["event:ok"]


def test_optional_pending_implementation_survives_failed_spec() -> None:
    class Plugin:
        @hookimpl(optionalhook=True)
        def maybe(self, actual):
            return f"optional:{actual}"

    class BadSpec:
        @hookspec
        def maybe(self, expected):
            pass

    class GoodSpec:
        @hookspec
        def maybe(self, actual):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    # Optional unknown hooks are allowed while no spec exists.
    pm.check_pending()

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    # The failed spec must not delete the pending implementation.
    pm.check_pending()

    pm.add_hookspecs(GoodSpec)
    assert pm.hook.maybe(actual="x") == ["optional:x"]


def test_existing_hookcaller_reference_survives_failed_spec_update() -> None:
    class Plugin:
        @hookimpl
        def target(self, value):
            return f"target:{value}"

    class BadSpec:
        @hookspec
        def target(self, wrong):
            pass

    class GoodSpec:
        @hookspec
        def target(self, value):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(Plugin(), name="plugin")

    hook_before = pm.hook.target

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    assert pm.hook.target is hook_before

    pm.add_hookspecs(GoodSpec)

    # Existing references to the HookCaller should observe the recovered
    # hook state. A rollback that replaces the HookCaller object will fail.
    assert hook_before(value="kept") == ["target:kept"]


def test_failed_spec_update_does_not_disturb_preexisting_valid_specs() -> None:
    class ExistingSpec:
        @hookspec
        def stable(self, value):
            pass

    class ExistingPlugin:
        @hookimpl
        def stable(self, value):
            return f"stable:{value}"

    class BadPlugin:
        @hookimpl
        def broken(self, extra):
            return extra

    class BadSpec:
        @hookspec
        def broken(self, value):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.add_hookspecs(ExistingSpec)
    pm.register(ExistingPlugin(), name="existing")

    assert pm.hook.stable(value="before") == ["stable:before"]

    pm.register(BadPlugin(), name="bad")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    assert pm.hook.stable(value="after") == ["stable:after"]


def test_hookimpl_order_is_preserved_across_failed_then_successful_spec() -> None:
    class FirstPlugin:
        @hookimpl(tryfirst=True)
        def target(self, value):
            return f"first:{value}"

    class NormalPlugin:
        @hookimpl
        def target(self, value):
            return f"normal:{value}"

    class LastPlugin:
        @hookimpl(trylast=True)
        def target(self, value):
            return f"last:{value}"

    class BadSpec:
        @hookspec
        def target(self, wrong):
            pass

    class GoodSpec:
        @hookspec
        def target(self, value):
            pass

    pm = pluggy.PluginManager(PROJECT)

    pm.register(LastPlugin(), name="last")
    pm.register(NormalPlugin(), name="normal")
    pm.register(FirstPlugin(), name="first")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    pm.add_hookspecs(GoodSpec)

    assert pm.hook.target(value="x") == [
        "first:x",
        "normal:x",
        "last:x",
    ]


def test_registering_more_pending_impls_after_failed_spec_uses_unknown_hook_path() -> None:
    class FirstPlugin:
        @hookimpl
        def target(self, actual):
            return f"first:{actual}"

    class SecondPlugin:
        @hookimpl
        def target(self, actual):
            return f"second:{actual}"

    class BadSpec:
        @hookspec
        def target(self, expected):
            pass

    class GoodSpec:
        @hookspec
        def target(self, actual):
            pass

    pm = pluggy.PluginManager(PROJECT)
    pm.register(FirstPlugin(), name="first")

    with pytest.raises(pluggy.PluginValidationError):
        pm.add_hookspecs(BadSpec)

    # Since the failed spec was not installed, registering another pending
    # implementation should not validate it against the rejected spec.
    pm.register(SecondPlugin(), name="second")

    assert_pending_unknown_hook(pm, "target")

    pm.add_hookspecs(GoodSpec)

    assert pm.hook.target(actual="ok") == [
        "second:ok",
        "first:ok",
    ]
```

