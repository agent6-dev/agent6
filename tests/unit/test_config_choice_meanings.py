from __future__ import annotations

import typing
from typing import Any

import pydantic

from agent6.config import model
from agent6.viewmodel import config_view


def test_every_literal_leaf_has_short_choice_meanings() -> None:
    """Every finite field, including mapping entries, owns a meaning for each value."""
    seen: set[type[pydantic.BaseModel]] = set()

    def walk(cls: type[pydantic.BaseModel]) -> None:
        if cls in seen:
            return
        seen.add(cls)
        for name, field in cls.model_fields.items():
            if field.exclude:
                continue
            extra = field.json_schema_extra
            meanings = extra.get("choice_meanings", {}) if isinstance(extra, dict) else {}

            def annotation(ann: Any, meanings: Any = meanings, name: str = name) -> None:
                if typing.get_origin(ann) is typing.Literal:
                    assert isinstance(meanings, dict)
                    for value in typing.get_args(ann):
                        meaning = meanings.get(str(value), "")
                        assert isinstance(meaning, str)
                        assert 2 <= len(meaning.split()) <= 10, (cls.__name__, name, value)
                    schema = config_view.field_schema(cls, [name])
                    assert schema is not None
                    assert schema.choice_meanings == meanings
                elif isinstance(ann, type) and issubclass(ann, pydantic.BaseModel):
                    walk(ann)
                else:
                    for arg in typing.get_args(ann):
                        annotation(arg)

            annotation(field.annotation)

    walk(model.Config)
    assert len(seen) > 20
