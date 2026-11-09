"""Model routes inherit each unset field from the required default."""

from __future__ import annotations

from typing import Any

import pydantic

from agent6 import kinds
from agent6.config import harness_config, validation


class ModelsConfig(pydantic.BaseModel):
    """The `[models]` table: a qualified model route per role.

    `Config.require_runnable` requires `default` before any model call.
    Each role inherits its unset fields directly from `default`.
    """

    model_config = validation.MODEL_CONFIG

    default: harness_config.RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "required route for every model call; set with `agent6 model default PROVIDER/MODEL`; "
            "the only fallback for each role's unset model, effort and temperature"
        ),
    )
    helper: harness_config.RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "model for the context summariser, gister (`compaction.elision_gists`), "
            "verify-command inference and prompt reviser; unset fields take `models.default`"
        ),
    )
    worker: harness_config.RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "model for `agent6 run`, `agent6 ask`, their resumes, machine agent states and the "
            "model-written squash message; unset fields take `models.default`"
        ),
    )
    reviewer: harness_config.RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "model for in-loop review panel seats without a route, the `sessions compare` and "
            "`--parallel` judge, and `sessions review`; unset fields take `models.default`"
        ),
    )
    planner: harness_config.RoleModel | None = pydantic.Field(
        default=None,
        description=(
            "model for `agent6 plan` and plan resumes; unset fields take `models.default`"
        ),
    )

    @pydantic.field_validator("default", "helper", "worker", "reviewer", "planner", mode="before")
    @classmethod
    def _parse_route(cls, data: Any, info: pydantic.ValidationInfo) -> Any:
        """Resolve a role's scalar route while accepting resolved objects internally.

        Args:
            data: The settings table or resolved role.
            info: The role's field name.

        Returns:
            The resolved role fields, or the object unchanged.

        Raises:
            ValueError: A split route.
            pydantic.ValidationError: An unqualified route.
        """
        validation.refuse_split_model(data, f"models.{info.field_name}")
        if data == {}:
            return None
        if isinstance(data, dict) and isinstance(data.get("model"), str):
            try:
                route = validation.model_route(data["model"])
            except ValueError as exc:
                # Writers validate the leaf even when a higher layer masks it.
                error = pydantic.ValidationError.from_exception_data(
                    cls.__name__,
                    [
                        {
                            "type": "value_error",
                            "loc": ("model",),
                            "input": data["model"],
                            "ctx": {"error": exc},
                        }
                    ],
                )
                raise error from exc
            return {**data, "provider": route.provider, "model": route.model}
        return data

    @pydantic.model_serializer(mode="wrap")
    def _serialize_settings(
        self, handler: pydantic.SerializerFunctionWrapHandler
    ) -> dict[str, Any]:
        """Keep unset fields absent so config copies preserve inheritance.

        Args:
            handler: Pydantic's serializer.

        Returns:
            Only the fields each role explicitly sets.
        """
        data = handler(self)
        for role in kinds.MODEL_ROLES:
            entry = getattr(self, role)
            if entry is not None and role in data:
                data[role] = {
                    field: value
                    for field, value in data[role].items()
                    if field in entry.model_fields_set
                }
        return data

    def configured(self) -> dict[str, harness_config.RoleModel]:
        """Return explicitly named routes with their effective sampling settings."""
        return {
            role: resolved
            for role in kinds.MODEL_ROLES
            if (entry := getattr(self, role)) is not None
            and entry.model
            and (resolved := self.resolve(role)) is not None
        }

    def resolve(self, role: kinds.RoleName) -> harness_config.RoleModel | None:
        """Resolve each unset field from `default`, never from another role.

        Args:
            role: The role.

        Returns:
            The resolved route and sampling settings, or None without a route.
        """
        base = self.default.model_dump(exclude_unset=True) if self.default is not None else {}
        entry = getattr(self, role)
        if entry is not None:
            base.update(entry.model_dump(exclude_unset=True))
        if not base.get("model"):
            return None
        route = validation.model_route(base["model"])
        return harness_config.RoleModel.model_validate(
            {**base, "provider": route.provider, "model": route.model}
        )

    def source_role(self, role: kinds.RoleName, field: str = "model") -> kinds.RoleName:
        """Return the role that supplies a leaf, so provenance names the key written.

        Args:
            role: The role.
            field: The leaf name.

        Returns:
            The role itself when the field is set, else `default`.
        """
        entry = getattr(self, role)
        return role if entry is not None and field in entry.model_fields_set else "default"
