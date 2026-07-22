from __future__ import annotations
import json
import re
import unicodedata
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any
import nltk
from nltk.metrics import edit_distance


@dataclass
class TranslationMemoryEntry:
    jp_text: str
    en_text: str
    page_number: int
    story_position: str
    region_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TranslationMemoryEntry:
        return cls(
            jp_text=data["jp_text"],
            en_text=data["en_text"],
            page_number=int(data.get("page_number", 0)),
            story_position=str(data.get("story_position", "")),
            region_index=int(data.get("region_index", 0)),
        )


class TranslationMemoryDB:
    """Core Translation Memory indexing, matching, and vocabulary engine."""

    def __init__(self) -> None:
        # Map manga_id -> list of TranslationMemoryEntry
        self.manga_entries: dict[str, list[TranslationMemoryEntry]] = {}
        # Map manga_id -> dict of bilingual word mappings (vocab)
        self.manga_vocab: dict[str, dict[str, str]] = {}

    def add_entry(self, manga_id: str, entry: TranslationMemoryEntry) -> None:
        """Add a translation entry to the memory. Idempotent."""
        if manga_id not in self.manga_entries:
            self.manga_entries[manga_id] = []
        
        # Check for exact duplicate to be idempotent
        for existing in self.manga_entries[manga_id]:
            if (existing.jp_text == entry.jp_text and
                existing.en_text == entry.en_text and
                existing.page_number == entry.page_number and
                existing.story_position == entry.story_position and
                existing.region_index == entry.region_index):
                return
        
        self.manga_entries[manga_id].append(entry)

    def load(self, path: Path | str) -> None:
        """Load database from JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"TM database not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        self.manga_entries = {}
        self.manga_vocab = data.get("vocabularies", {})
        
        db_entries = data.get("entries", {})
        for manga_id, entries_list in db_entries.items():
            self.manga_entries[manga_id] = [
                TranslationMemoryEntry.from_dict(e) for e in entries_list
            ]

    def save(self, path: Path | str) -> None:
        """Save database to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        db_entries = {
            manga_id: [e.to_dict() for e in entries]
            for manga_id, entries in self.manga_entries.items()
        }
        
        data = {
            "version": "1.0",
            "vocabularies": self.manga_vocab,
            "entries": db_entries,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def normalize_text(self, text: str) -> str:
        """Apply standard NFKC normalization and strip all whitespace."""
        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        return re.sub(r'\s+', '', normalized)

    def exact_match(self, manga_id: str, jp_query: str) -> str | None:
        """Look up exact translation match. local manga first, then global fallback."""
        normalized_query = self.normalize_text(jp_query)
        if not normalized_query:
            return None
        
        # 1. Search locally within manga_id
        if manga_id in self.manga_entries:
            for entry in self.manga_entries[manga_id]:
                if self.normalize_text(entry.jp_text) == normalized_query:
                    return entry.en_text
        
        # 2. Search globally across all other mangas
        for m_id, entries in self.manga_entries.items():
            if m_id == manga_id:
                continue
            for entry in entries:
                if self.normalize_text(entry.jp_text) == normalized_query:
                    return entry.en_text
                    
        return None

    def fuzzy_match(
        self, manga_id: str, jp_query: str, threshold: float = 0.8
    ) -> tuple[str, float] | None:
        """Match query using Jaccard unigram/bigram pre-filter and edit distance reranking."""
        normalized_query = self.normalize_text(jp_query)
        if not normalized_query:
            return None

        # Collect candidate entries
        candidates: list[tuple[str, TranslationMemoryEntry]] = []
        if manga_id in self.manga_entries:
            for entry in self.manga_entries[manga_id]:
                candidates.append((manga_id, entry))
        for m_id, entries in self.manga_entries.items():
            if m_id == manga_id:
                continue
            for entry in entries:
                candidates.append((m_id, entry))

        if not candidates:
            return None

        # Pre-filter using Jaccard similarity of character bigrams/unigrams to find top 20 candidates.
        query_tokens = self._get_char_ngrams(normalized_query)
        
        scored_candidates: list[tuple[TranslationMemoryEntry, float]] = []
        for c_manga_id, entry in candidates:
            entry_norm = self.normalize_text(entry.jp_text)
            entry_tokens = self._get_char_ngrams(entry_norm)
            
            intersection = len(query_tokens & entry_tokens)
            union = len(query_tokens | entry_tokens)
            jaccard = intersection / union if union > 0 else 0.0
            
            scored_candidates.append((entry, jaccard))

        # Sort and take top 20 candidates
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        top_candidates = scored_candidates[:20]

        best_entry: TranslationMemoryEntry | None = None
        best_sim = -1.0

        for entry, _ in top_candidates:
            entry_norm = self.normalize_text(entry.jp_text)
            dist = edit_distance(normalized_query, entry_norm)
            max_len = max(len(normalized_query), len(entry_norm))
            sim = 1.0 - (dist / max_len) if max_len > 0 else 0.0
            
            if sim > best_sim:
                best_sim = sim
                best_entry = entry

        if best_entry and best_sim >= threshold:
            return best_entry.en_text, best_sim

        return None

    def _get_char_ngrams(self, text: str) -> set[str]:
        unigrams = set(text)
        bigrams = {text[i:i+2] for i in range(len(text)-1)}
        return unigrams | bigrams

    def get_context(self, manga_id: str, story_position: str, limit: int = 5) -> list[tuple[str, str]]:
        """Retrieve the preceding dialogue sentences sorted chronologically."""
        if manga_id not in self.manga_entries:
            return []
        
        entries = self.manga_entries[manga_id]
        
        # Robust parsing of story_position to sort numerically by page and region
        def parse_story_pos(pos: str) -> tuple[int, int, str]:
            parts = pos.split("::")
            if len(parts) >= 3:
                try:
                    return (int(parts[1]), int(parts[2]), pos)
                except ValueError:
                    pass
            return (0, 0, pos)

        query_key = parse_story_pos(story_position)

        def get_sort_key(e: TranslationMemoryEntry) -> tuple[int, int, str]:
            if e.story_position:
                return parse_story_pos(e.story_position)
            return (e.page_number, e.region_index, "")

        sorted_entries = sorted(entries, key=get_sort_key)
        
        # Filter preceding entries
        preceding = []
        for e in sorted_entries:
            if get_sort_key(e) < query_key:
                preceding.append(e)
            else:
                break
        
        return [(e.jp_text, e.en_text) for e in preceding[-limit:]]

    def build_vocabularies(self, cooccurrence_threshold: float = 0.3) -> None:
        """Extract bilingual terminology and character names for each manga."""
        self.manga_vocab = {}
        for manga_id, entries in self.manga_entries.items():
            self.manga_vocab[manga_id] = self._extract_vocabulary_for_manga(entries, cooccurrence_threshold)

    def _extract_vocabulary_for_manga(
        self, entries: list[TranslationMemoryEntry], threshold: float
    ) -> dict[str, str]:
        # Offline stop words configuration
        stop_words = {
            "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", 
            "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", 
            "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", 
            "theirs", "themselves", "what", "which", "who", "whom", "this", "that", 
            "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", 
            "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", 
            "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", 
            "at", "by", "for", "with", "about", "against", "between", "into", "through", 
            "during", "before", "after", "above", "below", "to", "from", "up", "down", 
            "in", "out", "on", "off", "over", "under", "again", "further", "then", "once",
            "youre", "dont", "im", "its", "cant", "wont", "didnt", "cant", "that", "this"
        }
        
        jp_regex = re.compile(r'[\u30a0-\u30ff\u30fc]+|[\u4e00-\u9fff]+')
        en_regex = re.compile(r'[a-zA-Z]+')

        pair_tokens = []
        all_jp_tokens = set()
        all_en_tokens = set()

        for entry in entries:
            jp_norm = self.normalize_text(entry.jp_text)
            en_norm = entry.en_text.lower()

            jp_words = set(jp_regex.findall(jp_norm))
            en_words = {w for w in en_regex.findall(en_norm) if w not in stop_words and len(w) > 1}

            pair_tokens.append((jp_words, en_words))
            all_jp_tokens.update(jp_words)
            all_en_tokens.update(en_words)

        jp_counts = {j: 0 for j in all_jp_tokens}
        en_counts = {e: 0 for e in all_en_tokens}
        co_counts = {}

        for jp_words, en_words in pair_tokens:
            for j in jp_words:
                jp_counts[j] += 1
            for e in en_words:
                en_counts[e] += 1
            for j in jp_words:
                for e in en_words:
                    pair_key = (j, e)
                    co_counts[pair_key] = co_counts.get(pair_key, 0) + 1

        # Calculate Jaccard association scores
        vocab = {}
        best_mappings = {}  # jp_word -> (en_word, score)

        for (j, e), co_count in co_counts.items():
            denominator = jp_counts[j] + en_counts[e] - co_count
            jaccard = co_count / denominator if denominator > 0 else 0.0
            if jaccard >= threshold:
                if j not in best_mappings or jaccard > best_mappings[j][1]:
                    best_mappings[j] = (e, jaccard)

        for j, (e, score) in best_mappings.items():
            vocab[j] = e

        return vocab
