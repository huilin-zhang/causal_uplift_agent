"""Retrieval over short method cards in docs/knowledge/*.md (the RAG part).

Each markdown file is split at "## " headings; each section is one chunk.
TF-IDF with cosine similarity is enough for a few dozen chunks and needs no
embedding model or network. Every hit keeps its file and heading, so the
agent can cite where a statement came from.
Code guide: section "Knowledge retrieval" (sec:rag).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class Hit:
    source: str
    section: str
    text: str
    score: float

    def citation(self) -> str:
        return f"{self.source} > {self.section}"


def split_sections(path: Path) -> list[tuple[str, str]]:
    sections, title, lines = [], path.stem, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            if lines:
                sections.append((title, "\n".join(lines).strip()))
            title, lines = line[3:].strip(), []
        elif not line.startswith("# "):
            lines.append(line)
    if lines:
        sections.append((title, "\n".join(lines).strip()))
    return [(t, body) for t, body in sections if body]


class KnowledgeBase:
    def __init__(self, folder: Path):
        self.chunks: list[tuple[str, str, str]] = []
        for path in sorted(Path(folder).glob("*.md")):
            for section, body in split_sections(path):
                self.chunks.append((path.name, section, body))
        corpus = [f"{s} {b}" for _, s, b in self.chunks]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vectorizer.fit_transform(corpus) if corpus else None

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[Hit]:
        if self.matrix is None:
            return []
        sims = cosine_similarity(self.vectorizer.transform([query]), self.matrix).ravel()
        order = sims.argsort()[::-1][:k]
        return [
            Hit(self.chunks[i][0], self.chunks[i][1], self.chunks[i][2], float(sims[i]))
            for i in order
            if sims[i] >= min_score
        ]
