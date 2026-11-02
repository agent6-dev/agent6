"""The config page's paired provider and model editor."""

from __future__ import annotations

import pathlib
from typing import ClassVar

import textual
from textual import app, binding, containers, widgets

from agent6 import errors
from agent6.config import layer_data, notices, write
from agent6.models import cache, choices
from agent6.ui.tui import config_forms, forms


class ModelModal(config_forms.FormModal[None]):
    """Edit a role's provider and model without replacing its other settings."""

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
        self._provider = route.provider if route and route.provider in eff.config.providers else ""
        self._model = route.model if route and self._provider else ""
        self._generation = 0

    def compose(self) -> app.ComposeResult:
        """Yield the provider, open-text model, target and save control."""
        with containers.VerticalScroll(id="model-box"):
            yield widgets.Static(f"Model: {self._role}")
            yield widgets.Static("Provider", classes="edit-label")
            yield forms.ChoiceField(
                ("Choose a provider", *sorted(self._eff.config.providers)),
                self._provider or "Choose a provider",
                id="model-provider",
            )
            yield widgets.Static("Model", classes="edit-label")
            with containers.Vertical(id="model-field"):
                yield forms.TypeaheadField(
                    self._model, cache.cached_models(self._provider), id="model-name"
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
        """Focus the provider and fetch its model suggestions."""
        self.query_one("#model-provider").focus()
        self._load_models()

    @textual.on(forms.ChoiceField.Changed)
    async def _provider_changed(self, event: forms.ChoiceField.Changed) -> None:
        """Clear the model before listing a different provider."""
        if event.field.id != "model-provider" or event.field.value == self._provider:
            return
        self._provider = event.field.value
        self._generation += 1
        container = self.query_one("#model-field", containers.Vertical)
        await container.remove_children()
        await container.mount(forms.TypeaheadField("", [], id="model-name"))
        self._load_models()

    def _load_models(self) -> None:
        """Fetch the selected provider's models off the UI thread."""
        self._generation += 1
        generation = self._generation
        provider = self._provider
        status = self.query_one("#model-catalog", widgets.Static)
        if provider not in self._eff.config.providers:
            status.update("Choose a configured provider, or add one on the Config page.")
            return
        status.update("Loading models…")
        host = self.app

        def fetch() -> None:
            models = choices.provider_model_choices(self._eff.config, provider)
            host.call_from_thread(self._apply_models, generation, models)

        self.run_worker(fetch, thread=True)

    def _apply_models(self, generation: int, models: list[str]) -> None:
        """Ignore a listing for an older selection or a closed editor."""
        if not self.is_mounted or generation != self._generation:
            return
        self.query_one("#model-name", forms.TypeaheadField).set_suggestions(models)
        self.query_one("#model-catalog", widgets.Static).update(
            "Unlisted model ids are accepted."
            if models
            else "Catalog empty or unavailable. You can type a model id."
        )

    def action_save(self) -> None:
        """Save the pair, keeping an invalid form open with the writer's reason."""
        provider = self.query_one("#model-provider", forms.ChoiceField).value
        model = self.query_one("#model-name", forms.TypeaheadField).value.strip()
        if provider not in self._eff.config.providers or not model:
            self.notify("Provider and model are required.", severity="error")
            return
        to_repo = self.query_one("#model-target", forms.ChoiceField).index == 1
        section = f"models.{self._role}"
        try:
            err = write.set_config_leaves(
                self._repo, section, {"provider": provider, "model": model}, to_repo=to_repo
            )
        except errors.OperatorError as exc:
            err = str(exc)
        if err:
            self.notify(err, severity="error", timeout=8.0)
            return
        self.notify(
            notices.write_message(
                self._repo,
                [f"{section}.provider", f"{section}.model"],
                f"Set {section} = {provider}/{model}",
                to_repo=to_repo,
                config_paths=self._config_paths,
            )
        )
        self.dismiss(None)

    def action_cancel(self) -> None:
        """Close without writing."""
        self.dismiss(None)
