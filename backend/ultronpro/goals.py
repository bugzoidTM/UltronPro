from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from pathlib import Path
import json
import random
import time

from ultronpro import llm
import os


@dataclass
class Goal:
    id: int
    title: str
    description: str | None
    status: str
    priority: int


class GoalPlanner:
    """Goal creation & planning with proactive non-deterministic ambition."""

    def __init__(self, state_path: str | Path = "/app/data/ambition_state.json"):
        self.state_path = Path(state_path)
        self.cooldown_sec = 6 * 3600

    def _load_state(self) -> dict[str, Any]:
        try:
            if self.state_path.exists():
                d = json.loads(self.state_path.read_text())
                if isinstance(d, dict):
                    d.setdefault("last_ambition_at", 0)
                    d.setdefault("recent_titles", [])
                    return d
        except Exception:
            pass
        return {"last_ambition_at": 0, "recent_titles": []}

    def _save_state(self, st: dict[str, Any]):
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            st["recent_titles"] = list(dict.fromkeys(st.get("recent_titles") or []))[-80:]
            self.state_path.write_text(json.dumps(st, ensure_ascii=False, indent=2))
        except Exception:
            pass

    def _fallback_ambition(self) -> dict[str, Any]:
        pool = [
            ("Mapear história da Roma Antiga por fases e eventos", "Construir linha do tempo causal (República, Império, queda) com fontes e sínteses.", 6),
            ("Construir atlas de conceitos fundamentais de física", "Conectar leis, hipóteses e exemplos práticos para transferência entre domínios.", 5),
            ("Criar mapa comparativo de sistemas políticos históricos", "Relacionar instituições, estabilidade e causas de transformação em civilizações distintas.", 5),
            ("Modelar taxonomia de erros cognitivos em decisões", "Catalogar vieses e estratégias de mitigação para melhorar qualidade decisória.", 6),
            ("Mapear evolução da computação: de Turing à IA moderna", "Organizar marcos, ideias-chave e rupturas tecnológicas com evidências.", 5),
        ]
        t, d, p = random.choice(pool)
        return {"title": t, "description": d, "priority": p, "ambition": True}

    def _llm_ambition(self, experiences: list[dict[str, Any]], existing_titles: list[str]) -> dict[str, Any] | None:
        # default off to keep low-cost + avoid endpoint latency spikes
        if str(os.getenv("ULTRON_AMBITION_USE_LLM", "0")).strip().lower() not in ("1", "true", "yes"):
            return None
        ctx = "\n".join((e.get("text") or "")[:220] for e in experiences[-12:])
        prompt = f"""Generate ONE abstract, proactive long-horizon goal for an autonomous learning system.
Constraints:
- Must NOT depend on current conflicts.
- Must be broad and exploratory (e.g., mapping a domain deeply).
- Return ONLY JSON: {{"title":"...","description":"...","priority":1..7}}
Avoid near-duplicates of existing titles:
{existing_titles[:20]}
Context:\n{ctx[:2400]}"""
        try:
            raw = llm.complete(prompt, strategy='cheap', json_mode=True)
            d = json.loads(raw) if raw else {}
            if not isinstance(d, dict) or not d.get("title"):
                return None
            return {
                "title": str(d.get("title") or "").strip()[:160],
                "description": str(d.get("description") or "").strip()[:500],
                "priority": int(max(1, min(7, int(d.get("priority") or 5)))),
                "ambition": True,
            }
        except Exception:
            return None

    def _novel_enough(self, title: str, existing_titles: list[str], recent_titles: list[str]) -> bool:
        t = (title or "").strip().lower()
        if not t or len(t) < 12:
            return False
        bag = [(x or "").strip().lower() for x in (existing_titles + recent_titles)]
        for e in bag:
            if not e:
                continue
            if t == e or t in e or e in t:
                return False
            # token overlap quick check
            ts = set([w for w in t.split() if len(w) >= 4])
            es = set([w for w in e.split() if len(w) >= 4])
            if ts and es and (len(ts & es) / max(1, len(ts | es))) >= 0.72:
                return False
        return True

    def propose_ambition(self, experiences: list[dict[str, Any]], existing_titles: list[str]) -> dict[str, Any] | None:
        st = self._load_state()
        now = time.time()
        if (now - float(st.get("last_ambition_at") or 0)) < self.cooldown_sec:
            return None

        candidate = self._llm_ambition(experiences, existing_titles) or self._fallback_ambition()
        if not self._novel_enough(candidate.get("title") or "", existing_titles, st.get("recent_titles") or []):
            # one retry fallback
            candidate = self._fallback_ambition()
            if not self._novel_enough(candidate.get("title") or "", existing_titles, st.get("recent_titles") or []):
                return None

        st["last_ambition_at"] = now
        st["recent_titles"] = (st.get("recent_titles") or []) + [candidate.get("title")]
        self._save_state(st)
        return candidate

    def build_weekly_milestones(self, goal_title: str, goal_description: str | None = None, weeks: int = 4) -> list[dict[str, Any]]:
        weeks = max(2, min(8, int(weeks)))
        prompt = f"""Break this long-horizon goal into {weeks} weekly milestones.
Return ONLY JSON array, each item:
{{"week_index":1..{weeks},"title":"...","progress_criteria":"..."}}
Goal: {goal_title}
Description: {goal_description or ''}
"""
        try:
            raw = llm.complete(prompt, strategy='cheap', json_mode=True)
            arr = json.loads(raw) if raw else []
            out = []
            if isinstance(arr, list):
                for i, m in enumerate(arr[:weeks], start=1):
                    if isinstance(m, dict) and m.get("title"):
                        out.append({
                            "week_index": int(max(1, min(weeks, int(m.get("week_index") or i)))),
                            "title": str(m.get("title") or "").strip()[:180],
                            "progress_criteria": str(m.get("progress_criteria") or "").strip()[:400],
                        })
            if out:
                out.sort(key=lambda x: x["week_index"])
                return out
        except Exception:
            pass

        # fallback determinístico
        base = (goal_title or "objetivo").strip()
        return [
            {"week_index": 1, "title": f"Escopo e mapa inicial: {base}", "progress_criteria": "Definir 10-20 subtemas e fontes prioritárias."},
            {"week_index": 2, "title": f"Coleta estruturada: {base}", "progress_criteria": "Registrar evidências e sínteses por subtema."},
            {"week_index": 3, "title": f"Integração causal: {base}", "progress_criteria": "Conectar relações, resolver conflitos e lacunas."},
            {"week_index": 4, "title": f"Consolidação e avaliação: {base}", "progress_criteria": "Produzir visão consolidada e checklist de qualidade."},
        ][:weeks]

    def propose_goals(self, experiences: list[dict[str, Any]], existing_goals: list[dict[str, Any]] | None = None) -> list[dict[str, Any]]:
        text = "\n".join((e.get("text") or "") for e in experiences[-10:]).lower()

        goals = []
        if "ultron" in text or "agi" in text or "curios" in text:
            goals.append({
                "title": "Construir núcleo de curiosidade + síntese (tese↔antítese↔síntese)",
                "description": "Manter perguntas úteis e resolver contradições com evidência.",
                "priority": 5,
            })
            goals.append({
                "title": "Melhorar ingestão multimodal (imagens/áudio) e extração",
                "description": "Aceitar anexos e registrar metadados; extrair texto quando possível.",
                "priority": 4,
            })

        goals.append({
            "title": "Aprender com tutores: registrar conhecimento como triplas com evidência",
            "description": "Aumentar confiança por suporte, reduzir por contradição.",
            "priority": 4,
        })

        existing_titles = [str(g.get("title") or "") for g in (existing_goals or [])]
        ambition = self.propose_ambition(experiences, existing_titles=existing_titles)
        if ambition:
            goals.append(ambition)

        seen = set()
        out = []
        for g in goals:
            t = g["title"].strip()
            if t in seen:
                continue
            seen.add(t)
            out.append(g)
        return out[:7]
