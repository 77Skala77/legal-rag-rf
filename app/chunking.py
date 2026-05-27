import re
from dataclasses import dataclass


@dataclass(frozen=True)
class LegalChunk:
    text: str
    chunk_index: int
    article_number: str | None = None
    section_title: str | None = None


@dataclass(frozen=True)
class _Segment:
    text: str
    article_number: str | None
    section_title: str | None


ARTICLE_RE = re.compile(
    r"^\s*Статья\s+(?P<number>\d+(?:\.\d+)*)\b", re.IGNORECASE
)
SECTION_RE = re.compile(r"^\s*(?P<kind>Раздел|Глава)\s+(?P<title>.+)$", re.IGNORECASE)
SUBUNIT_RE = re.compile(
    r"^\s*(?:Пункт\s+\d+(?:\.\d+)*|Абзац\s+\d+|\d+(?:\.\d+)*[.)])(?:\s|$)",
    re.IGNORECASE,
)


def chunk_legal_text(
    text: str, max_chars: int = 1800, overlap: int = 200
) -> list[LegalChunk]:
    """Split legal text along structural boundaries, falling back to overlap chunks."""
    normalized = _normalize(text)
    if not normalized:
        return []
    _validate_limits(max_chars, overlap)

    segments, has_structure = _parse_structural_segments(normalized)
    if not has_structure:
        return [
            LegalChunk(text=part, chunk_index=index)
            for index, part in enumerate(_split_with_overlap(normalized, max_chars, overlap))
        ]

    packed: list[_Segment] = []
    for segment in segments:
        if len(segment.text) > max_chars:
            packed.extend(
                _Segment(part, segment.article_number, segment.section_title)
                for part in _split_with_overlap(segment.text, max_chars, overlap)
            )
            continue

        previous = packed[-1] if packed else None
        can_merge = (
            previous is not None
            and previous.article_number == segment.article_number
            and previous.section_title == segment.section_title
            and len(previous.text) + 2 + len(segment.text) <= max_chars
        )
        if can_merge:
            packed[-1] = _Segment(
                f"{previous.text}\n\n{segment.text}",
                previous.article_number,
                previous.section_title,
            )
        else:
            packed.append(segment)

    return [
        LegalChunk(
            text=segment.text,
            chunk_index=index,
            article_number=segment.article_number,
            section_title=segment.section_title,
        )
        for index, segment in enumerate(packed)
    ]


def _parse_structural_segments(text: str) -> tuple[list[_Segment], bool]:
    segments: list[_Segment] = []
    lines: list[str] = []
    current_article: str | None = None
    current_section: str | None = None
    has_structure = False

    def flush() -> None:
        nonlocal lines
        segment_text = "\n".join(lines).strip()
        if segment_text:
            segments.append(_Segment(segment_text, current_article, current_section))
        lines = []

    for line in text.splitlines():
        stripped = line.strip()
        article_match = ARTICLE_RE.match(stripped)
        section_match = SECTION_RE.match(stripped)
        is_subunit = SUBUNIT_RE.match(stripped) is not None

        if article_match or section_match or is_subunit:
            has_structure = True
            flush()
            if section_match:
                current_section = stripped
                current_article = None
            elif article_match:
                current_article = article_match.group("number")
            lines.append(line)
        else:
            lines.append(line)
    flush()
    return segments, has_structure


def _split_with_overlap(text: str, max_chars: int, overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + max_chars, length)
        if end < length:
            lower_bound = start + max_chars // 2
            newline_cut = text.rfind("\n", lower_bound, end)
            space_cut = text.rfind(" ", lower_bound, end)
            preferred_cut = max(newline_cut, space_cut)
            if preferred_cut > start:
                end = preferred_cut
        part = text[start:end].strip()
        if part:
            chunks.append(part)
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def _normalize(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _validate_limits(max_chars: int, overlap: int) -> None:
    if max_chars < 1:
        raise ValueError("max_chars must be positive.")
    if overlap < 0 or overlap >= max_chars:
        raise ValueError("overlap must be non-negative and smaller than max_chars.")
