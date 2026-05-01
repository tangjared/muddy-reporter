from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .models import Report, SourceDoc


def render_html(report: Report, sources: list[SourceDoc], template_dir: str) -> str:
    env = Environment(
        loader=FileSystemLoader(template_dir),
        autoescape=select_autoescape(["html", "xml"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    tpl = env.get_template("report.html.j2")
    return tpl.render(report=report.model_dump(), sources=[s.model_dump() for s in sources])


def write_html(out_path: str, html: str) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html, encoding="utf-8")

