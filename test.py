# Make registry updates independent of prior crawling

A `Registry` lazily discovers resource identifiers, subresources, and anchors when it is crawled or queried.

Updating a registry after discovery has occurred must not leave it with resolution results belonging to resources that have since been replaced. The observable contents of a registry should depend on its current registrations and their precedence, not on whether earlier versions happened to be crawled.

Update registry mutation and combination behavior so that replacing resources remains consistent with lazy discovery.

## Required behavior

* Replacing a resource registered at an existing URI must make subsequent resource and anchor resolution reflect the replacement resource.
* Resources and anchors discoverable only through a displaced resource must no longer be available after the resulting registry is crawled.
* Resources and anchors introduced by the replacement must be discoverable normally.
* A registry produced by replacing resources after an earlier crawl must be observationally equivalent to a registry built from the same current registrations without crawling the displaced resources first.
* Existing registrations unrelated to the replaced resource must remain available.
* A resource or anchor must remain available when it is still independently registered or still discoverable through another unaffected registered resource.
* Replacement behavior must correctly account for internal identifiers, canonical identifiers, relative identifiers, nested subresources, and the existing empty-fragment URI normalization.
* When the same URI occurs more than once in one update operation, the final registration for that URI must determine the resulting resource graph.
* All existing resource-addition paths must remain consistent, including:

  * `with_resource`
  * `with_resources`
  * `with_contents`
  * resource insertion using `resource @ registry`
* `combine()` must preserve its existing registry precedence rules, but the resources and anchors visible in the combined registry must correspond to the registrations that win according to those rules. Discoveries belonging only to displaced registrations must not leak into the result.
* Registry immutability must be preserved. Updating or combining registries must not modify any input registry.
* Resource addition and combination must remain lazy. They must not crawl resources or invoke retrieval merely to perform the update.
* Repeatedly crawling an unchanged registry must remain idempotent and must not repeat discovery work.
* Existing retrieval behavior, exception types, resolver behavior, and public APIs must remain compatible.
* Do not require callers to manually remove resources or construct a new registry in order to replace an existing registration.

Resource lookup, anchor lookup, resolver lookup, iteration, and registry size must all reflect the same current resource graph after crawling.


```python
from __future__ import annotations

from collections.abc import Iterable

import pytest

from referencing import Anchor, Registry, Resource, Specification, exceptions
from referencing.jsonschema import DRAFT202012


def _id_of(contents: object) -> str | None:
    if not isinstance(contents, dict):
        return None

    identifier = contents.get("id")
    return identifier if isinstance(identifier, str) else None


def _subresources_of(contents: object) -> Iterable[object]:
    if not isinstance(contents, dict):
        return ()

    children = contents.get("children", ())
    return children if isinstance(children, list) else ()


def _anchors_in(
    specification: Specification[object],
    contents: object,
) -> Iterable[Anchor[object]]:
    if not isinstance(contents, dict):
        return ()

    anchors = contents.get("anchors", {})
    if not isinstance(anchors, dict):
        return ()

    return (
        Anchor(
            name=name,
            resource=specification.create_resource(value),
        )
        for name, value in anchors.items()
    )


GRAPH = Specification(
    name="replacement-graph",
    id_of=_id_of,
    subresources_of=_subresources_of,
    anchors_in=_anchors_in,
    maybe_in_subresource=lambda segments, resolver, subresource: resolver,
)


def graph_resource(contents: object) -> Resource[object]:
    return GRAPH.create_resource(contents)


def assert_no_resource(
    registry: Registry[object],
    uri: str,
) -> None:
    with pytest.raises(exceptions.NoSuchResource):
        registry.contents(uri)


def assert_no_anchor(
    registry: Registry[object],
    uri: str,
    name: str,
) -> None:
    with pytest.raises(
        (exceptions.NoSuchAnchor, exceptions.NoSuchResource),
    ):
        registry.anchor(uri, name)


# ---------------------------------------------------------------------------
# PUBLIC TEST
# ---------------------------------------------------------------------------


def test_replacing_crawled_resource_drops_old_anchor() -> None:
    uri = "https://example.test/root"

    old = graph_resource(
        {
            "anchors": {
                "old": {"version": "old"},
            },
        },
    )
    new = graph_resource(
        {
            "anchors": {
                "new": {"version": "new"},
            },
        },
    )

    original = Registry().with_resource(uri, old).crawl()
    replaced = original.with_resource(uri, new).crawl()

    # Registry snapshots remain immutable.
    assert original.anchor(uri, "old").value.resource.contents == {
        "version": "old",
    }

    assert replaced.contents(uri) == new.contents
    assert_no_anchor(replaced, uri, "old")

    assert replaced.anchor(uri, "new").value.resource.contents == {
        "version": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: nested resource graph retraction
# ---------------------------------------------------------------------------


def test_replacement_retracts_nested_relative_resources_and_anchors() -> None:
    root_uri = "https://example.test/schemas/root"
    child_uri = "https://example.test/schemas/child"
    grandchild_uri = (
        "https://example.test/schemas/nested/grandchild"
    )

    old = graph_resource(
        {
            "children": [
                {
                    "id": "child",
                    "anchors": {
                        "child-anchor": {"from": "child"},
                    },
                    "children": [
                        {
                            "id": "nested/grandchild",
                            "anchors": {
                                "deep": {"from": "grandchild"},
                            },
                        },
                    ],
                },
            ],
        },
    )
    new = graph_resource(
        {
            "anchors": {
                "current": {"from": "new"},
            },
        },
    )

    replaced = (
        Registry()
        .with_resource(root_uri, old)
        .crawl()
        .with_resource(root_uri, new)
        .crawl()
    )

    assert_no_resource(replaced, child_uri)
    assert_no_resource(replaced, grandchild_uri)
    assert_no_anchor(replaced, child_uri, "child-anchor")
    assert_no_anchor(replaced, grandchild_uri, "deep")

    assert replaced.anchor(
        root_uri,
        "current",
    ).value.resource.contents == {
        "from": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: registration URI versus canonical resource identifier
# ---------------------------------------------------------------------------


def test_replacement_retracts_old_canonical_identifier() -> None:
    registered_uri = "https://retrieval.test/slot"
    old_canonical = "https://schemas.test/old"
    new_canonical = "https://schemas.test/new"

    old = graph_resource(
        {
            "id": old_canonical,
            "anchors": {
                "version": {"value": "old"},
            },
        },
    )
    new = graph_resource(
        {
            "id": new_canonical,
            "anchors": {
                "version": {"value": "new"},
            },
        },
    )

    replaced = (
        Registry()
        .with_resource(registered_uri, old)
        .crawl()
        .with_resource(registered_uri, new)
        .crawl()
    )

    assert replaced.contents(registered_uri) == new.contents
    assert replaced.contents(new_canonical) == new.contents

    assert_no_resource(replaced, old_canonical)
    assert_no_anchor(replaced, old_canonical, "version")

    assert replaced.anchor(
        new_canonical,
        "version",
    ).value.resource.contents == {
        "value": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: directly registered resources must survive graph cleanup
# ---------------------------------------------------------------------------


def test_explicit_resource_at_old_child_uri_is_preserved() -> None:
    root_uri = "https://example.test/root"
    shared_uri = "https://example.test/shared"

    old_root = graph_resource(
        {
            "children": [
                {
                    "id": shared_uri,
                    "anchors": {
                        "derived": {"source": "old-root"},
                    },
                },
            ],
        },
    )
    explicit = graph_resource(
        {
            "id": shared_uri,
            "anchors": {
                "explicit": {"source": "explicit"},
            },
        },
    )
    replacement_root = graph_resource({})

    registry = Registry().with_resource(root_uri, old_root).crawl()

    # This registration replaces the discovered child as the current
    # resource at shared_uri.
    registry = registry.with_resource(shared_uri, explicit).crawl()

    replaced = registry.with_resource(
        root_uri,
        replacement_root,
    ).crawl()

    assert replaced.contents(shared_uri) == explicit.contents

    assert replaced.anchor(
        shared_uri,
        "explicit",
    ).value.resource.contents == {
        "source": "explicit",
    }

    assert_no_anchor(replaced, shared_uri, "derived")


# ---------------------------------------------------------------------------
# HIDDEN: an independently reachable discovery must survive
# Preservation test: current implementation may already pass.
# ---------------------------------------------------------------------------


def test_discovery_reachable_from_another_root_is_preserved() -> None:
    first_uri = "https://example.test/first"
    second_uri = "https://example.test/second"
    shared_uri = "https://example.test/shared"

    shared = {
        "id": shared_uri,
        "anchors": {
            "kept": {"source": "shared"},
        },
    }

    first = graph_resource({"children": [shared]})
    second = graph_resource({"children": [shared]})
    replacement = graph_resource({})

    registry = (
        Registry()
        .with_resources(
            [
                (first_uri, first),
                (second_uri, second),
            ],
        )
        .crawl()
    )

    replaced = registry.with_resource(
        first_uri,
        replacement,
    ).crawl()

    assert replaced.contents(shared_uri) == shared
    assert replaced.anchor(
        shared_uri,
        "kept",
    ).value.resource.contents == {
        "source": "shared",
    }


# ---------------------------------------------------------------------------
# HIDDEN: one resource registered through multiple aliases
# Preservation test: rejects URI-wide over-deletion.
# ---------------------------------------------------------------------------


def test_replacing_one_alias_preserves_other_alias_reachability() -> None:
    alias_a = "https://retrieval.test/a"
    alias_b = "https://retrieval.test/b"
    canonical = "https://schemas.test/shared"

    shared = graph_resource(
        {
            "id": canonical,
            "anchors": {
                "kept": {"source": "shared"},
            },
        },
    )

    registry = (
        Registry()
        .with_resources(
            [
                (alias_a, shared),
                (alias_b, shared),
            ],
        )
        .crawl()
    )

    replaced = registry.with_resource(
        alias_a,
        graph_resource({}),
    ).crawl()

    assert replaced.contents(alias_b) == shared.contents
    assert replaced.contents(canonical) == shared.contents

    assert replaced.anchor(
        canonical,
        "kept",
    ).value.resource.contents == {
        "source": "shared",
    }


# ---------------------------------------------------------------------------
# HIDDEN: duplicate normalized URIs in one batch
# ---------------------------------------------------------------------------


def test_batch_replacement_uses_only_final_resource_graph() -> None:
    uri = "https://example.test/root"
    old_child = "https://example.test/old-child"
    middle_child = "https://example.test/middle-child"
    final_child = "https://example.test/final-child"

    def version(
        name: str,
        child: str,
    ) -> Resource[object]:
        return graph_resource(
            {
                "anchors": {
                    name: {"version": name},
                },
                "children": [
                    {
                        "id": child,
                        "value": name,
                    },
                ],
            },
        )

    old = version("old", old_child)
    middle = version("middle", middle_child)
    final = version("final", final_child)

    registry = Registry().with_resource(uri, old).crawl()

    replaced = registry.with_resources(
        [
            (uri + "#", middle),
            (uri, final),
        ],
    ).crawl()

    assert replaced.contents(uri) == final.contents
    assert replaced.contents(final_child) == {
        "id": final_child,
        "value": "final",
    }

    assert_no_resource(replaced, old_child)
    assert_no_resource(replaced, middle_child)
    assert_no_anchor(replaced, uri, "old")
    assert_no_anchor(replaced, uri, "middle")

    assert replaced.anchor(
        uri,
        "final",
    ).value.resource.contents == {
        "version": "final",
    }


# ---------------------------------------------------------------------------
# HIDDEN: combine with an uncrawled winning registration
# ---------------------------------------------------------------------------


def test_combine_uncrawled_winner_retracts_loser_graph() -> None:
    uri = "https://example.test/root"
    old_child = "https://example.test/old-child"
    new_child = "https://example.test/new-child"

    old = graph_resource(
        {
            "anchors": {
                "old": {"version": "old"},
            },
            "children": [
                {
                    "id": old_child,
                    "value": "old",
                },
            ],
        },
    )
    new = graph_resource(
        {
            "anchors": {
                "new": {"version": "new"},
            },
            "children": [
                {
                    "id": new_child,
                    "value": "new",
                },
            ],
        },
    )

    crawled_loser = Registry().with_resource(uri, old).crawl()
    uncrawled_winner = Registry().with_resource(uri, new)

    combined = crawled_loser.combine(
        uncrawled_winner,
    ).crawl()

    assert combined.contents(uri) == new.contents
    assert combined.contents(new_child) == {
        "id": new_child,
        "value": "new",
    }

    assert_no_resource(combined, old_child)
    assert_no_anchor(combined, uri, "old")

    assert combined.anchor(
        uri,
        "new",
    ).value.resource.contents == {
        "version": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: both combine inputs have already been crawled
# ---------------------------------------------------------------------------


def test_combine_crawled_winner_drops_loser_only_anchors() -> None:
    uri = "https://example.test/root"

    old = graph_resource(
        {
            "anchors": {
                "old": {"version": "old"},
            },
        },
    )
    new = graph_resource(
        {
            "anchors": {
                "new": {"version": "new"},
            },
        },
    )

    loser = Registry().with_resource(uri, old).crawl()
    winner = Registry().with_resource(uri, new).crawl()

    combined = loser.combine(winner)

    assert combined.contents(uri) == new.contents
    assert_no_anchor(combined, uri, "old")

    assert combined.anchor(
        uri,
        "new",
    ).value.resource.contents == {
        "version": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: Resource.__matmul__ uses a separate insertion path
# ---------------------------------------------------------------------------


def test_matmul_replacement_retracts_previous_graph() -> None:
    uri = "https://example.test/root"
    old_child = "https://example.test/old-child"

    old = graph_resource(
        {
            "id": uri,
            "anchors": {
                "old": {"version": "old"},
            },
            "children": [
                {
                    "id": old_child,
                    "value": "old",
                },
            ],
        },
    )
    new = graph_resource(
        {
            "id": uri,
            "anchors": {
                "new": {"version": "new"},
            },
        },
    )

    original = (old @ Registry()).crawl()
    replaced = (new @ original).crawl()

    assert replaced.contents(uri) == new.contents
    assert_no_resource(replaced, old_child)
    assert_no_anchor(replaced, uri, "old")

    assert replaced.anchor(
        uri,
        "new",
    ).value.resource.contents == {
        "version": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: use the real JSON Schema specification and Resolver API
# Also exercises with_contents().
# ---------------------------------------------------------------------------


def test_json_schema_resolver_cannot_see_replaced_subschema() -> None:
    registered_uri = "https://retrieval.test/root"
    old_child_uri = "https://schemas.test/child"
    new_child_uri = "https://schemas.test/replacement"

    old = {
        "$id": "https://schemas.test/root",
        "$defs": {
            "child": {
                "$id": old_child_uri,
                "$anchor": "old",
                "const": "old",
            },
        },
    }
    new = {
        "$id": "https://schemas.test/new-root",
        "$defs": {
            "child": {
                "$id": new_child_uri,
                "$anchor": "new",
                "const": "new",
            },
        },
    }

    replaced = (
        Registry()
        .with_contents(
            [(registered_uri, old)],
            default_specification=DRAFT202012,
        )
        .crawl()
        .with_contents(
            [(registered_uri, new)],
            default_specification=DRAFT202012,
        )
        .crawl()
    )

    resolver = replaced.resolver()

    with pytest.raises(exceptions.Unresolvable):
        resolver.lookup(old_child_uri + "#old")

    assert resolver.lookup(
        new_child_uri + "#new",
    ).contents == {
        "$id": new_child_uri,
        "$anchor": "new",
        "const": "new",
    }


# ---------------------------------------------------------------------------
# HIDDEN: prevent an eager full rebuild and repeated crawl work
# ---------------------------------------------------------------------------


def test_replacement_remains_lazy_and_crawl_is_idempotent() -> None:
    calls = {
        "anchors": 0,
        "children": 0,
    }

    def children(contents: object) -> Iterable[object]:
        calls["children"] += 1
        return _subresources_of(contents)

    def anchors(
        specification: Specification[object],
        contents: object,
    ) -> Iterable[Anchor[object]]:
        calls["anchors"] += 1
        return _anchors_in(specification, contents)

    counted = Specification(
        name="counted",
        id_of=_id_of,
        subresources_of=children,
        anchors_in=anchors,
        maybe_in_subresource=(
            lambda segments, resolver, subresource: resolver
        ),
    )

    uri = "https://example.test/root"

    old = counted.create_resource(
        {
            "anchors": {
                "old": {},
            },
        },
    )
    new = counted.create_resource(
        {
            "anchors": {
                "new": {},
            },
        },
    )

    original = Registry().with_resource(uri, old).crawl()

    calls["anchors"] = 0
    calls["children"] = 0

    pending = original.with_resource(uri, new)

    # Replacement itself must remain lazy.
    assert calls == {
        "anchors": 0,
        "children": 0,
    }

    once = pending.crawl()
    after_first_crawl = dict(calls)

    assert after_first_crawl["anchors"] > 0
    assert after_first_crawl["children"] > 0

    twice = once.crawl()

    # An already crawled registry must not be traversed again.
    assert calls == after_first_crawl
    assert once == twice

    assert_no_anchor(twice, uri, "old")
    assert twice.anchor(
        uri,
        "new",
    ).value.resource.contents == {}
```
