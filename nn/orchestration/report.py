import yaml
from dataclasses import asdict, dataclass


@dataclass
class ReportMeta:
    archetype: str
    version: int
    parent: str | None
    spec_hash: str
    hypothesis: str
    holdout_acc: float
    holdout_loss: float
    per_class: dict
    decision: str
    strike: int
    next_hypothesis: str | None


def parse_report(md: str) -> ReportMeta:
    parts = md.split("---", 2)
    if len(parts) < 3:
        raise ValueError("report has no '---' frontmatter fences")
    front = yaml.safe_load(parts[1]) or {}
    required = set(ReportMeta.__dataclass_fields__)
    missing = required - set(front)
    if missing:
        raise ValueError(f"missing report keys: {sorted(missing)}")
    return ReportMeta(**{k: front[k] for k in required})


def render_report(meta: ReportMeta, body: str) -> str:
    front = yaml.safe_dump(asdict(meta))
    return "---\n" + front + "---\n\n" + body
