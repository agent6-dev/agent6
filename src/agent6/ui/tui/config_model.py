"""The config page's qualified model editor."""

from __future__ import annotations

import pathlib
from typing import ClassVar

from textual import app, binding, containers, widgets

from agent6 import errors
from agent6.config import layer_data, notices, write
from agent6.models import choices
from agent6.ui.tui import config_forms, forms


class ModelModal(config_forms.FormModal[None]):
    """Edit a role's qualified model without replacing its sampling settings."""

    BINDINGS: ClassVar = [binding.Binding("escape", "cancel", "Cancel")]
    CSS = (
        forms.FORM_CSS
        + """
    ModelModal { align: center middle; }
    #model-box {
        width: 80; height: auto;
        border: round $accent; padding: 1 2; background: $surface;
    }
    #model-field, #model-actions { height: auto; }
    #model-actions { padding-top: 1; }
    """
    )

    def __init__(
        self,
        repo_root: pathlib.Path,
        eff: layer_data.EffectiveConfig,
        role: str,
        source: str,
        config_paths: tuple[pathlib.Path, ...] = (),
    ) -> None:
        super().__init__()
        self._repo = repo_root
        self._eff = eff
        self._role = role
        self._source = source
        self._config_paths = config_paths
        route = eff.config.models.configured().get(role)
        self._model = f"{route.provider}/{route.model}" if route else ""

    def compose(self) -> app.ComposeResult:
        """Yield the qualified model, target and save control."""
        with containers.VerticalScroll(id="model-box"):
            yield widgets.Static(f"Model: {self._role}")
            yield widgets.Static("Model", classes="edit-label")
            with containers.Vertical(id="model-field"):
                yield forms.TypeaheadField(
                    self._model, choices.route_choices(self._eff.config), id="model-name"
                )
            yield widgets.Static("", id="model-catalog", classes="edit-label")
            yield widgets.Static("save to", classes="edit-label")
            yield forms.ChoiceField(
                ("global config", "repo config"),
                "repo config" if self._source == "repo" else "global config",
                id="model-target",
            )
            with containers.Horizontal(id="model-actions"):
                yield forms.ActionItem("Save", "save")
                yield forms.ActionItem("Cancel", "cancel")

    def on_mount(self) -> None:
        """Focus the route and fetch model suggestions off the UI thread."""
        self.query_one("#model-name").focus()
        self.query_one("#model-catalog", widgets.Static).update("Loading models…")
        host = self.app

        def fetch() -> None:
            models = choices.config_value_choices(self._eff, f"models.{self._role}.model")
            host.call_from_thread(self._apply_models, models)

        self.run_worker(fetch, thread=True)

    def _apply_models(self, models: list[str]) -> None:
        """Ignore a listing for a closed editor."""
        if not self.is_mounted:
            return
        self.query_one("#model-name", forms.TypeaheadField).set_suggestions(models)
        self.query_one("#model-catalog", widgets.Static).update(
            "Unlisted provider/model routes are accepted."
            if models
            else "Catalog empty or unavailable. Type a provider/model route."
        )

    def action_save(self) -> None:
        """Save the route, keeping an invalid form open with the writer's reason."""
        model = self.query_one("#model-name", forms.TypeaheadField).value
        to_repo = self.query_one("#model-target", forms.ChoiceField).index == 1
        section = f"models.{self._role}"
        try:
            err = write.set_config_leaves(self._repo, section, {"model": model}, to_repo=to_repo)
        except errors.OperatorError as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
            return
        self.notify(
            notices.write_message(
                self._repo,
                [f"{section}.model"],
                f"Set {section}.model = {model}",
                to_repo=to_repo,
                config_paths=self._config_paths,
            )
        )
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close without writing."""
        self.dismiss(None)
