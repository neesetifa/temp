```python
from __future__ import annotations

import pytest

import tomlkit
from tomlkit import exceptions


# ---------------------------------------------------------------------------
# PUBLIC TEST
# ---------------------------------------------------------------------------


def test_rename_parsed_key_preserves_position_and_trivia() -> None:
    source = (
        "# project metadata\n"
        "old_name   =   \"demo\"  # keep this comment\n"
        "version = \"1.0\"\n"
    )

    doc = tomlkit.parse(source)

    doc.rename("old_name", "name")

    assert doc.as_string() == (
        "# project metadata\n"
        "name   =   \"demo\"  # keep this comment\n"
        "version = \"1.0\"\n"
    )

    assert "old_name" not in doc
    assert doc["name"] == "demo"
    assert doc["version"] == "1.0"
    assert doc.unwrap() == {
        "name": "demo",
        "version": "1.0",
    }


# ---------------------------------------------------------------------------
# HIDDEN TESTS
# ---------------------------------------------------------------------------


def test_rename_missing_key_raises_and_leaves_document_unchanged() -> None:
    source = (
        "name = \"demo\"\n"
        "\n"
        "# dependency block follows\n"
        "[tool]\n"
        "enabled = true\n"
    )

    doc = tomlkit.parse(source)
    before = doc.as_string()
    before_unwrapped = doc.unwrap()

    with pytest.raises(exceptions.NonExistentKey):
        doc.rename("missing", "renamed")

    assert doc.as_string() == before
    assert doc.unwrap() == before_unwrapped
    assert doc["name"] == "demo"
    assert doc["tool"]["enabled"] is True


def test_rename_existing_target_raises_and_rolls_back() -> None:
    source = (
        "# first value\n"
        "alpha = 1\n"
        "beta = 2  # target already exists\n"
        "gamma = 3\n"
    )

    doc = tomlkit.parse(source)
    before = doc.as_string()
    before_items = list(doc.items())
    before_unwrapped = doc.unwrap()

    with pytest.raises(exceptions.KeyAlreadyPresent):
        doc.rename("alpha", "beta")

    assert doc.as_string() == before
    assert list(doc.items()) == before_items
    assert doc.unwrap() == before_unwrapped
    assert doc["alpha"] == 1
    assert doc["beta"] == 2
    assert doc["gamma"] == 3


def test_rename_programmatically_created_document_key() -> None:
    doc = tomlkit.document()
    doc.add(tomlkit.comment("generated file"))
    doc.add("old", 1)
    doc.add("after", 2)

    doc.rename("old", "new")

    assert doc.as_string() == (
        "# generated file\n"
        "new = 1\n"
        "after = 2\n"
    )
    assert "old" not in doc
    assert doc["new"] == 1
    assert doc.unwrap() == {
        "new": 1,
        "after": 2,
    }


def test_rename_regular_table_preserves_header_position_and_contents() -> None:
    source = (
        "# package table\n"
        "[tool.old]  # table comment\n"
        "name = \"demo\"  # inline value comment\n"
        "enabled = true\n"
        "\n"
        "[tool.other]\n"
        "name = \"keep\"\n"
    )

    doc = tomlkit.parse(source)

    doc["tool"].rename("old", "new")

    assert doc.as_string() == (
        "# package table\n"
        "[tool.new]  # table comment\n"
        "name = \"demo\"  # inline value comment\n"
        "enabled = true\n"
        "\n"
        "[tool.other]\n"
        "name = \"keep\"\n"
    )

    assert "old" not in doc["tool"]
    assert doc["tool"]["new"]["name"] == "demo"
    assert doc["tool"]["new"]["enabled"] is True
    assert doc["tool"]["other"]["name"] == "keep"

    assert doc.unwrap() == {
        "tool": {
            "new": {
                "name": "demo",
                "enabled": True,
            },
            "other": {
                "name": "keep",
            },
        },
    }


def test_rename_only_changes_the_selected_table_branch() -> None:
    source = (
        "[alpha.old]\n"
        "value = 1\n"
        "\n"
        "[beta.old]\n"
        "value = 2\n"
        "\n"
        "[alpha.keep]\n"
        "value = 3\n"
    )

    doc = tomlkit.parse(source)

    doc["alpha"].rename("old", "renamed")

    assert doc.as_string() == (
        "[alpha.renamed]\n"
        "value = 1\n"
        "\n"
        "[beta.old]\n"
        "value = 2\n"
        "\n"
        "[alpha.keep]\n"
        "value = 3\n"
    )

    assert "old" not in doc["alpha"]
    assert doc["alpha"]["renamed"]["value"] == 1
    assert doc["beta"]["old"]["value"] == 2
    assert doc["alpha"]["keep"]["value"] == 3


def test_rename_array_of_tables_updates_each_header_without_reordering() -> None:
    source = (
        "# first package\n"
        "[[tool.old]]\n"
        "name = \"first\"\n"
        "\n"
        "# second package\n"
        "[[tool.old]]\n"
        "name = \"second\"\n"
        "\n"
        "[tool.keep]\n"
        "name = \"unchanged\"\n"
    )

    doc = tomlkit.parse(source)

    doc["tool"].rename("old", "new")

    assert doc.as_string() == (
        "# first package\n"
        "[[tool.new]]\n"
        "name = \"first\"\n"
        "\n"
        "# second package\n"
        "[[tool.new]]\n"
        "name = \"second\"\n"
        "\n"
        "[tool.keep]\n"
        "name = \"unchanged\"\n"
    )

    assert "old" not in doc["tool"]
    assert len(doc["tool"]["new"]) == 2
    assert doc["tool"]["new"][0]["name"] == "first"
    assert doc["tool"]["new"][1]["name"] == "second"
    assert doc["tool"]["keep"]["name"] == "unchanged"

    assert doc.unwrap() == {
        "tool": {
            "new": [
                {
                    "name": "first",
                },
                {
                    "name": "second",
                },
            ],
            "keep": {
                "name": "unchanged",
            },
        },
    }


def test_rename_dotted_key_proxy_updates_rendered_key_and_lookup() -> None:
    source = (
        "server.old = \"blue\"\n"
        "server.keep = \"gray\"\n"
        "\n"
        "[other]\n"
        "old = \"untouched\"\n"
    )

    doc = tomlkit.parse(source)

    doc["server"].rename("old", "new")

    assert doc.as_string() == (
        "server.new = \"blue\"\n"
        "server.keep = \"gray\"\n"
        "\n"
        "[other]\n"
        "old = \"untouched\"\n"
    )

    assert "old" not in doc["server"]
    assert doc["server"]["new"] == "blue"
    assert doc["server"]["keep"] == "gray"
    assert doc["other"]["old"] == "untouched"

    assert doc.unwrap() == {
        "server": {
            "new": "blue",
            "keep": "gray",
        },
        "other": {
            "old": "untouched",
        },
    }


def test_rename_quoted_key_uses_existing_key_rendering_rules() -> None:
    source = (
        "\"old key\" = \"value\"\n"
        "plain = true\n"
    )

    doc = tomlkit.parse(source)

    doc.rename("old key", "new key")

    assert doc.as_string() == (
        "\"new key\" = \"value\"\n"
        "plain = true\n"
    )

    assert "old key" not in doc
    assert doc["new key"] == "value"
    assert doc["plain"] is True


def test_rename_table_to_existing_table_rolls_back_nested_state() -> None:
    source = (
        "[tool.old]\n"
        "name = \"old\"\n"
        "flag = true\n"
        "\n"
        "[tool.new]\n"
        "name = \"existing\"\n"
        "\n"
        "[tool.after]\n"
        "name = \"after\"\n"
    )

    doc = tomlkit.parse(source)
    before = doc.as_string()
    before_unwrapped = doc.unwrap()

    with pytest.raises(exceptions.KeyAlreadyPresent):
        doc["tool"].rename("old", "new")

    assert doc.as_string() == before
    assert doc.unwrap() == before_unwrapped

    assert doc["tool"]["old"]["name"] == "old"
    assert doc["tool"]["old"]["flag"] is True
    assert doc["tool"]["new"]["name"] == "existing"
    assert doc["tool"]["after"]["name"] == "after"


def test_rename_preserves_table_object_and_value_identity() -> None:
    source = (
        "[tool.old]\n"
        "items = [\"a\", \"b\"]\n"
        "\n"
        "[tool.after]\n"
        "items = []\n"
    )

    doc = tomlkit.parse(source)
    old_table = doc["tool"]["old"]
    old_items = old_table["items"]

    doc["tool"].rename("old", "renamed")

    assert doc["tool"]["renamed"] is old_table
    assert doc["tool"]["renamed"]["items"] is old_items
    assert doc["tool"]["renamed"]["items"] == ["a", "b"]

    assert doc.as_string() == (
        "[tool.renamed]\n"
        "items = [\"a\", \"b\"]\n"
        "\n"
        "[tool.after]\n"
        "items = []\n"
    )
```

