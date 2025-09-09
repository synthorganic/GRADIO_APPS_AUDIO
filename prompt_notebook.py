import csv
from pathlib import Path

try:
    import gradio as gr  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    class _DummyGradio:
        @staticmethod
        def update(**kwargs):
            return kwargs

    gr = _DummyGradio()  # type: ignore

NOTEBOOK_PATH = Path(__file__).resolve().parent / "prompt_notebook.csv"


def _read_prompts() -> list[str]:
    """Return list of prompts stored in the CSV notebook."""
    if NOTEBOOK_PATH.exists():
        with NOTEBOOK_PATH.open("r", newline="") as f:
            for row in csv.reader(f):
                return [p for p in row if p]
    return []


def save_prompt(prompt: str) -> None:
    """Append a prompt to the CSV notebook if it is not empty or duplicated."""
    if not prompt:
        return
    prompts = _read_prompts()
    if prompt not in prompts:
        prompts.append(prompt)
        with NOTEBOOK_PATH.open("w", newline="") as f:
            csv.writer(f).writerow(prompts)


def show_notebook():
    """Return update dict to display the notebook dropdown with current prompts."""
    return gr.update(choices=_read_prompts(), visible=True)


def load_prompt(value: str):
    """Return update dict to populate a prompt textbox with the chosen value."""
    return gr.update(value=value)
