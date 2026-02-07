"""Curiosidade Adaptativa - Meta-learning para gerar perguntas melhores.

O CuriosityProcessor agora:
1. Mantém templates de perguntas com scores de eficácia
2. Aprende quais templates geram respostas úteis (longas, com triplas)
3. Gera perguntas contextuais baseadas em experiências recentes
4. Prioriza lacunas no conhecimento (conceitos mencionados mas não definidos)
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class QuestionTemplate:
    """Template para gerar perguntas."""
    id: str
    pattern: str  # com placeholders {concept}, {context}
    category: str  # definition, example, clarification, evidence, comparison
    base_priority: int = 3
    success_count: int = 0
    failure_count: int = 0
    total_triples_generated: int = 0
    
    @property
    def effectiveness_score(self) -> float:
        """Score de eficácia baseado em histórico."""
        total = self.success_count + self.failure_count
        if total < 3:
            return 0.5  # Prior neutro
        # Wilson score lower bound (simplificado)
        p = self.success_count / total
        z = 1.96  # 95% confidence
        n = total
        denominator = 1 + z*z/n
        center = p + z*z/(2*n)
        spread = z * ((p*(1-p) + z*z/(4*n))/n)**0.5
        return (center - spread) / denominator

    @property
    def adjusted_priority(self) -> float:
        """Prioridade ajustada pela eficácia."""
        return self.base_priority * (0.5 + self.effectiveness_score)


# Templates iniciais - serão refinados com o tempo
DEFAULT_TEMPLATES = [
    # Definições
    QuestionTemplate("def_what", "O que é {concept}?", "definition", 5),
    QuestionTemplate("def_meaning", "Qual o significado de {concept}?", "definition", 4),
    QuestionTemplate("def_explain", "Explique {concept} em termos simples.", "definition", 4),
    
    # Exemplos
    QuestionTemplate("ex_concrete", "Dê 2 exemplos concretos de {concept}.", "example", 4),
    QuestionTemplate("ex_realworld", "Como {concept} aparece no mundo real?", "example", 3),
    QuestionTemplate("ex_application", "Onde {concept} é aplicado na prática?", "example", 3),
    
    # Clarificação
    QuestionTemplate("clar_difference", "Qual a diferença entre {concept} e conceitos similares?", "clarification", 3),
    QuestionTemplate("clar_not", "O que {concept} NÃO é? (contra-exemplos)", "clarification", 3),
    QuestionTemplate("clar_confusion", "Quais são confusões comuns sobre {concept}?", "clarification", 2),
    
    # Evidência
    QuestionTemplate("ev_source", "Qual a fonte/origem de {concept}?", "evidence", 3),
    QuestionTemplate("ev_verify", "Como verificar se {concept} é verdade?", "evidence", 4),
    QuestionTemplate("ev_consensus", "Existe consenso científico sobre {concept}?", "evidence", 3),
    
    # Comparação
    QuestionTemplate("cmp_relate", "Como {concept} se relaciona com {context}?", "comparison", 3),
    QuestionTemplate("cmp_cause", "O que causa {concept}? Quais as consequências?", "comparison", 4),
    
    # Meta-aprendizado
    QuestionTemplate("meta_learn", "O que mais devo saber sobre {concept}?", "meta", 2),
    QuestionTemplate("meta_important", "Por que {concept} é importante?", "meta", 3),
]


@dataclass
class Question:
    question: str
    context: str | None = None
    priority: int = 0
    template_id: str | None = None
    concept: str | None = None


class CuriosityProcessor:
    """Processador de Curiosidade Adaptativo.
    
    Aprende quais perguntas geram respostas úteis e ajusta sua estratégia.
    """
    
    def __init__(self, state_path: str | Path = "/app/data/curiosity_state.json"):
        self.state_path = Path(state_path)
        self.templates: dict[str, QuestionTemplate] = {}
        self.concept_mentions: dict[str, int] = {}  # concept -> mention count
        self.concept_defined: set[str] = set()  # concepts que já foram definidos
        self._load_state()
    
    def _load_state(self):
        """Carrega estado persistido."""
        # Inicializa com defaults
        for t in DEFAULT_TEMPLATES:
            self.templates[t.id] = QuestionTemplate(
                id=t.id,
                pattern=t.pattern,
                category=t.category,
                base_priority=t.base_priority,
            )
        
        # Sobrescreve com estado salvo
        if self.state_path.exists():
            try:
                data = json.loads(self.state_path.read_text())
                for tid, stats in data.get("templates", {}).items():
                    if tid in self.templates:
                        self.templates[tid].success_count = stats.get("success", 0)
                        self.templates[tid].failure_count = stats.get("failure", 0)
                        self.templates[tid].total_triples_generated = stats.get("triples", 0)
                self.concept_mentions = data.get("concept_mentions", {})
                self.concept_defined = set(data.get("concept_defined", []))
            except Exception:
                pass
    
    def _save_state(self):
        """Persiste estado."""
        try:
            self.state_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "templates": {
                    tid: {
                        "success": t.success_count,
                        "failure": t.failure_count,
                        "triples": t.total_triples_generated,
                    }
                    for tid, t in self.templates.items()
                },
                "concept_mentions": dict(list(self.concept_mentions.items())[-500:]),  # keep last 500
                "concept_defined": list(self.concept_defined)[-500:],
            }
            self.state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        except Exception:
            pass
    
    def record_answer_feedback(
        self,
        template_id: str | None,
        concept: str | None,
        answer_length: int,
        triples_extracted: int,
    ):
        """Registra feedback sobre uma resposta para ajustar scores."""
        if not template_id or template_id not in self.templates:
            return
        
        t = self.templates[template_id]
        
        # Heurística: resposta útil se >= 50 chars e >= 1 tripla
        is_useful = answer_length >= 50 and triples_extracted >= 1
        
        if is_useful:
            t.success_count += 1
            t.total_triples_generated += triples_extracted
        else:
            t.failure_count += 1
        
        # Marca conceito como definido se teve resposta útil
        if concept and is_useful:
            self.concept_defined.add(concept.lower().strip())
        
        self._save_state()
    
    def _extract_concepts(self, text: str) -> list[str]:
        """Extrai conceitos mencionados no texto."""
        if not text:
            return []
        
        # Capitalizado ou entre aspas = provável conceito
        concepts = []
        
        # Palavras capitalizadas (exceto início de frase)
        words = text.split()
        for i, w in enumerate(words):
            clean = re.sub(r'[^\w]', '', w)
            if len(clean) >= 3 and clean[0].isupper() and i > 0:
                # Não é início de frase
                prev = words[i-1] if i > 0 else ""
                if not prev.endswith(('.', '!', '?', ':')):
                    concepts.append(clean)
        
        # Entre aspas
        quoted = re.findall(r'["\']([^"\']{2,30})["\']', text)
        concepts.extend(quoted)
        
        # Termos técnicos comuns
        tech_patterns = [
            r'\b(AGI|IA|AI|ML|NLP|API|P2P)\b',
            r'\b\w+(?:ção|mento|dade|ismo|logia)\b',  # sufixos técnicos PT
        ]
        for pat in tech_patterns:
            matches = re.findall(pat, text, re.IGNORECASE)
            concepts.extend(matches)
        
        # Dedupe e limpa
        seen = set()
        result = []
        for c in concepts:
            key = c.lower().strip()
            if key not in seen and len(key) >= 3:
                seen.add(key)
                result.append(c.strip())
        
        return result[:20]
    
    def _find_knowledge_gaps(self, experiences: list[dict]) -> list[str]:
        """Encontra conceitos mencionados mas não definidos."""
        # Atualiza contagem de menções
        for e in experiences:
            text = e.get("text", "")
            for concept in self._extract_concepts(text):
                key = concept.lower()
                self.concept_mentions[key] = self.concept_mentions.get(key, 0) + 1
        
        # Gaps: mencionados >= 2x mas não definidos
        gaps = []
        for concept, count in sorted(
            self.concept_mentions.items(),
            key=lambda x: -x[1]
        ):
            if count >= 2 and concept not in self.concept_defined:
                gaps.append(concept)
                if len(gaps) >= 10:
                    break
        
        return gaps
    
    def _select_template(self, category: str | None = None) -> QuestionTemplate:
        """Seleciona template usando Thompson Sampling simplificado."""
        candidates = list(self.templates.values())
        if category:
            candidates = [t for t in candidates if t.category == category]
        if not candidates:
            candidates = list(self.templates.values())
        
        # Thompson-ish: sample from Beta(success+1, failure+1)
        def sample_score(t: QuestionTemplate) -> float:
            import random
            # Aproximação: média + ruído proporcional à incerteza
            total = t.success_count + t.failure_count + 2
            mean = (t.success_count + 1) / total
            uncertainty = 1 / (total ** 0.5)
            return mean + random.gauss(0, uncertainty) + t.base_priority * 0.1
        
        return max(candidates, key=sample_score)
    
    def _generate_question(
        self,
        template: QuestionTemplate,
        concept: str,
        context: str | None = None,
    ) -> Question:
        """Gera pergunta a partir de template."""
        question = template.pattern.replace("{concept}", concept)
        if "{context}" in question and context:
            question = question.replace("{context}", context[:100])
        elif "{context}" in question:
            question = question.replace("{context}", "outros tópicos relacionados")
        
        return Question(
            question=question,
            context=context,
            priority=int(template.adjusted_priority),
            template_id=template.id,
            concept=concept,
        )

    def propose(
        self,
        experiences: list[dict[str, Any]],
        open_questions: list[str],
        target_count: int = 3,
    ) -> list[dict[str, Any]]:
        """Propõe perguntas adaptativas baseadas no contexto."""
        open_set = set(q.strip().lower() for q in open_questions)
        recent_text = "\n".join(e.get("text", "") for e in experiences[-10:])
        
        candidates: list[Question] = []
        
        # 1. Perguntas sobre lacunas de conhecimento
        gaps = self._find_knowledge_gaps(experiences)
        for gap in gaps[:3]:
            template = self._select_template("definition")
            q = self._generate_question(template, gap, recent_text[:200])
            candidates.append(q)
        
        # 2. Perguntas sobre conceitos recentes
        recent_concepts = self._extract_concepts(recent_text)
        for concept in recent_concepts[:5]:
            if concept.lower() in self.concept_defined:
                # Já definido: pedir exemplos ou evidências
                category = random.choice(["example", "evidence", "comparison"])
            else:
                category = "definition"
            
            template = self._select_template(category)
            q = self._generate_question(template, concept, recent_text[:200])
            candidates.append(q)
        
        # 3. Perguntas bootstrap se não houver experiências
        if not experiences or len(candidates) < target_count:
            bootstrap = [
                Question(
                    "Qual é o objetivo imediato do UltronPRO (o que ele precisa conseguir fazer primeiro)?",
                    priority=5,
                    template_id="bootstrap",
                ),
                Question(
                    "Quais tipos de entrada devo tratar como 'experiência' válida?",
                    priority=4,
                    template_id="bootstrap",
                ),
                Question(
                    "Como devo verificar se aprendi algo corretamente?",
                    priority=4,
                    template_id="bootstrap",
                ),
            ]
            candidates.extend(bootstrap)
        
        # Dedupe e filtra
        out: list[dict[str, Any]] = []
        seen = set()
        for q in sorted(candidates, key=lambda x: -x.priority):
            key = q.question.strip().lower()
            if key in open_set or key in seen:
                continue
            if q.question.startswith("```") or '"question"' in q.question:
                continue
            
            seen.add(key)
            out.append({
                "question": q.question,
                "context": q.context,
                "priority": q.priority,
                "template_id": q.template_id,
                "concept": q.concept,
            })
            
            if len(out) >= target_count:
                break
        
        self._save_state()
        return out
    
    def get_stats(self) -> dict[str, Any]:
        """Retorna estatísticas do sistema de curiosidade."""
        templates_stats = []
        for t in sorted(self.templates.values(), key=lambda x: -x.effectiveness_score):
            templates_stats.append({
                "id": t.id,
                "category": t.category,
                "pattern": t.pattern,
                "success": t.success_count,
                "failure": t.failure_count,
                "triples": t.total_triples_generated,
                "effectiveness": round(t.effectiveness_score, 3),
            })
        
        return {
            "total_templates": len(self.templates),
            "concepts_tracked": len(self.concept_mentions),
            "concepts_defined": len(self.concept_defined),
            "top_gaps": self._find_knowledge_gaps([])[:5],
            "templates": templates_stats,
        }
