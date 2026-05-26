from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    label: str
    regions: tuple[str, ...]
    default_selected: bool = False
    experimental: bool = False
    mode: str = "on-demand"


CATALOG: tuple[ModelSpec, ...] = (
    ModelSpec(
        model_id="openai.gpt-oss-20b",
        family="openai",
        label="OpenAI gpt-oss-20b",
        regions=("ap-osaka-1", "eu-frankfurt-1", "us-chicago-1"),
        default_selected=True,
    ),
    ModelSpec(
        model_id="openai.gpt-oss-120b",
        family="openai",
        label="OpenAI gpt-oss-120b",
        regions=("ap-osaka-1", "eu-frankfurt-1", "us-chicago-1"),
    ),
    ModelSpec(
        model_id="google.gemini-2.5-flash",
        family="gemini",
        label="Google Gemini 2.5 Flash",
        regions=("ap-osaka-1", "ap-hyderabad-1", "eu-frankfurt-1", "us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
        default_selected=True,
    ),
    ModelSpec(
        model_id="google.gemini-2.5-pro",
        family="gemini",
        label="Google Gemini 2.5 Pro",
        regions=("ap-osaka-1", "eu-frankfurt-1", "us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
    ),
    ModelSpec(
        model_id="google.gemini-2.5-flash-lite",
        family="gemini",
        label="Google Gemini 2.5 Flash-Lite",
        regions=("eu-frankfurt-1", "us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
    ),
    ModelSpec(
        model_id="xai.grok-3",
        family="grok",
        label="xAI Grok 3",
        regions=("us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="xai.grok-3-mini",
        family="grok",
        label="xAI Grok 3 Mini",
        regions=("us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="xai.grok-4",
        family="grok",
        label="xAI Grok 4",
        regions=("us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="xai.grok-4.3",
        family="grok",
        label="xAI Grok 4.3",
        regions=("us-ashburn-1", "us-chicago-1", "us-phoenix-1"),
        default_selected=True,
        experimental=True,
    ),
    ModelSpec(
        model_id="meta.llama-3.3-70b-instruct",
        family="meta",
        label="Meta Llama 3.3 70B",
        regions=("ap-osaka-1", "us-chicago-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="meta.llama-3.3-70b-instruct-fp8-dynamic",
        family="meta",
        label="Meta Llama 3.3 70B Dynamic FP8",
        regions=("ap-osaka-1", "us-chicago-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="meta.llama-3.2-90b-vision-instruct",
        family="meta",
        label="Meta Llama 3.2 90B Vision",
        regions=("ap-osaka-1", "us-chicago-1"),
        experimental=True,
    ),
    ModelSpec(
        model_id="meta.llama-4-scout-17b-16e-instruct",
        family="meta",
        label="Meta Llama 4 Scout 17B 16E Instruct",
        regions=("ap-osaka-1", "us-chicago-1"),
        default_selected=True,
        experimental=True,
    ),
    ModelSpec(
        model_id="cohere.command-a-03-2025",
        family="cohere",
        label="Cohere Command A 03-2025",
        regions=("us-chicago-1",),
        default_selected=True,
        experimental=True,
    ),
)

DEFAULT_FAMILIES: tuple[str, ...] = ("openai", "gemini")


def get_family_names() -> list[str]:
    return sorted({model.family for model in CATALOG})


def get_model(model_id: str) -> ModelSpec | None:
    for model in CATALOG:
        if model.model_id == model_id:
            return model
    return None


def resolve_models(
    requested_families: list[str] | None,
    requested_models: list[str] | None,
    include_experimental: bool,
) -> list[ModelSpec]:
    explicit_models = requested_models or []
    explicit_specs: list[ModelSpec] = []
    for model_id in explicit_models:
        spec = get_model(model_id)
        if spec is None:
            raise SystemExit(f"Unknown model: {model_id}. Use --list-models to see supported values.")
        explicit_specs.append(spec)

    requested_family_list = requested_families or list(DEFAULT_FAMILIES)
    unknown_families = sorted(set(requested_family_list) - set(get_family_names()))
    if unknown_families:
        joined = ", ".join(unknown_families)
        raise SystemExit(f"Unknown family: {joined}. Use --list-families to see supported values.")

    family_specs: list[ModelSpec] = []
    for family in requested_family_list:
        matching = [model for model in CATALOG if model.family == family and model.default_selected]
        if not include_experimental:
            matching = [model for model in matching if not model.experimental]
        family_specs.extend(matching)

    ordered: list[ModelSpec] = []
    seen = set()
    for spec in [*family_specs, *explicit_specs]:
        if not include_experimental and spec.experimental and spec.model_id not in explicit_models:
            continue
        if spec.model_id not in seen:
            ordered.append(spec)
            seen.add(spec.model_id)

    if not ordered:
        raise SystemExit(
            "No models selected. Use --list-models to inspect the catalog, "
            "or pass --model explicitly for non-default families."
        )
    return ordered


def format_model_listing() -> str:
    lines = []
    for model in CATALOG:
        default_marker = " default" if model.default_selected else ""
        experimental_marker = " experimental" if model.experimental else ""
        regions = ", ".join(model.regions)
        lines.append(
            f"{model.model_id} [{model.family}] ({model.mode}){default_marker}{experimental_marker} :: {regions}"
        )
    return "\n".join(lines)
