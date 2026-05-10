"""Text processing utilities.

Shared helper functions for text normalization, formatting,
conversation history processing, input sanitization, and
skill alias resolution.
"""

from __future__ import annotations

import re


def format_conversation_history(messages: list[dict[str, str]]) -> str:
    """Format a list of messages into a readable conversation string.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.

    Returns:
        Formatted multi-line string of the conversation.
    """
    lines: list[str] = []
    for msg in messages:
        role = msg.get("role", "unknown").upper()
        content = msg.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)



# Skill Alias System


# Canonical skill name → all aliases that should match catalog entries.
# The FIRST entry is the canonical form; all entries are lowercased for matching.
SKILL_ALIAS_MAP: dict[str, list[str]] = {
    "java": ["java", "core java", "java ee", "j2ee", "enterprise java", "java 8",
             "java platform", "java web services", "java frameworks", "java design patterns",
             "enterprise java beans", "spring"],
    "spring": ["spring", "spring boot", "spring framework", "java frameworks"],
    "sql": ["sql", "sql server", "mysql", "postgresql", "oracle", "pl/sql", "t-sql",
            "database", "relational database"],
    "python": ["python", "django", "flask", "pandas", "numpy"],
    "javascript": ["javascript", "js", "node.js", "nodejs", "react", "angular", "vue"],
    "typescript": ["typescript", "ts"],
    "angular": ["angular", "angularjs", "angular 6"],
    "react": ["react", "reactjs", "react.js"],
    "aws": ["aws", "amazon web services", "cloud computing", "amazon"],
    "docker": ["docker", "container", "containerization"],
    "kubernetes": ["kubernetes", "k8s"],
    "html": ["html", "html5"],
    "css": ["css", "css3", "stylesheet"],
    "rest": ["rest", "rest api", "restful", "web services"],
    "c#": ["c#", "csharp", "c sharp", ".net", "dotnet", "asp.net"],
    "c++": ["c++", "cpp"],
    "excel": ["excel", "ms excel", "microsoft excel", "spreadsheet"],
    "word": ["word", "ms word", "microsoft word"],
    "powerpoint": ["powerpoint", "ms powerpoint", "microsoft powerpoint"],
    "salesforce": ["salesforce", "sfdc", "crm"],
    "sap": ["sap", "sap abap", "sap hana"],
    "machine learning": ["machine learning", "ml", "deep learning", "dl",
                         "artificial intelligence", "ai", "data science"],
    "data science": ["data science", "data analysis", "analytics", "statistics",
                     "basic statistics"],
    "networking": ["networking", "network", "tcp/ip", "ccna", "cisco"],
    "linux": ["linux", "unix", "shell scripting", "bash"],
    "devops": ["devops", "ci/cd", "jenkins", "gitlab"],
    "php": ["php"],
    "ruby": ["ruby", "ruby on rails", "rails"],
    "scala": ["scala"],
    "go": ["go", "golang"],
    "swift": ["swift", "ios"],
    "accounting": ["accounting", "financial accounting", "bookkeeping"],
    "finance": ["finance", "financial", "banking"],
}

# Build a reverse lookup: alias → canonical name
_ALIAS_TO_CANONICAL: dict[str, str] = {}
for canonical, aliases in SKILL_ALIAS_MAP.items():
    for alias in aliases:
        _ALIAS_TO_CANONICAL[alias.lower()] = canonical


def normalize_skill_name(skill: str) -> str:
    """Normalize a skill name for consistent matching.

    Lowercases, strips whitespace, and handles common variations.

    Args:
        skill: Raw skill name from user input.

    Returns:
        Normalized skill string.
    """
    normalized = skill.strip().lower()
    # Handle common abbreviations
    abbreviations: dict[str, str] = {
        "js": "javascript",
        "ts": "typescript",
        "py": "python",
        "aws": "amazon web services",
        "gcp": "google cloud",
        "k8s": "kubernetes",
        "db": "database",
        "sql": "sql",
        "nosql": "nosql",
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "dl": "deep learning",
        "nlp": "natural language processing",
        "devops": "devops",
        "ci/cd": "ci/cd",
    }
    return abbreviations.get(normalized, normalized)


def get_skill_aliases(skill: str) -> list[str]:
    """Get all known aliases for a skill, including the skill itself.

    Looks up the canonical form, then returns all aliases for that
    canonical skill. Falls back to just the normalized skill if
    no aliases are found.

    Args:
        skill: Raw skill name.

    Returns:
        List of all known aliases (lowercased).
    """
    normalized = skill.strip().lower()
    # Check if this is a known alias
    canonical = _ALIAS_TO_CANONICAL.get(normalized)
    if canonical:
        return SKILL_ALIAS_MAP[canonical]
    # Check if it's a canonical name itself
    if normalized in SKILL_ALIAS_MAP:
        return SKILL_ALIAS_MAP[normalized]
    # Fallback: return just the skill itself
    return [normalized]


def parse_duration_minutes(duration_str: str) -> int | None:
    """Extract numeric minutes from a duration string.

    Handles formats like '30 minutes', 'max 20', 'Variable', etc.

    Args:
        duration_str: Raw duration string from catalog.

    Returns:
        Integer minutes, or None if unparseable.
    """
    if not duration_str or duration_str.lower() in ("", "variable", "untimed", "-"):
        return None

    match = re.search(r"(\d+)", duration_str)
    if match:
        return int(match.group(1))
    return None


def truncate_text(text: str, max_length: int = 200) -> str:
    """Truncate text to a maximum length with ellipsis.

    Args:
        text: Input text.
        max_length: Maximum character count.

    Returns:
        Truncated text with '...' if exceeded.
    """
    if len(text) <= max_length:
        return text
    return text[: max_length - 3] + "..."


# Input Sanitization

# Maximum characters per message
MAX_MESSAGE_LENGTH = 2000

# Minimum ratio of unique tokens to total tokens before we consider it noise
NOISE_UNIQUE_RATIO = 0.15


def sanitize_input(text: str) -> str:
    """Sanitize user input to prevent timeouts and injection.

    - Strips HTML/script tags
    - Strips SQL injection artifacts (OR, AND, semicolons, comments)
    - Strips JavaScript function calls (alert, prompt, etc.)
    - Compresses repeated tokens
    - Truncates to MAX_MESSAGE_LENGTH
    - Collapses excessive whitespace

    Args:
        text: Raw user input.

    Returns:
        Sanitized text string.
    """
    if not text:
        return text

    # 1. Strip HTML/script tags
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Strip SQL injection-like patterns (but keep the meaningful words)
    text = re.sub(r"\b(DROP\s+TABLE|DELETE\s+FROM|INSERT\s+INTO|UPDATE\s+SET)\b",
                  " ", text, flags=re.IGNORECASE)

    # 2b. Strip broader SQL injection artifacts
    text = re.sub(r"\b(OR|AND)\s+\d+=\d+", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"--", " ", text)
    text = re.sub(r";", " ", text)
    text = re.sub(r"'", " ", text)

    # 2c. Strip JavaScript function calls like alert(), prompt(), confirm()
    text = re.sub(r"\b(alert|prompt|confirm|eval|document|window)\s*\([^)]*\)",
                  " ", text, flags=re.IGNORECASE)

    # 3. Compress repeated tokens
    text = _compress_repeated_tokens(text)

    # 4. Collapse excessive whitespace
    text = re.sub(r"\s+", " ", text).strip()

    # 5. Truncate
    if len(text) > MAX_MESSAGE_LENGTH:
        text = text[:MAX_MESSAGE_LENGTH]

    return text


def _compress_repeated_tokens(text: str) -> str:
    """Detect and compress extremely repetitive text.

    If the same token appears more than 5 times, keep only 2 occurrences.

    Args:
        text: Input text.

    Returns:
        Compressed text.
    """
    tokens = text.split()
    if len(tokens) < 10:
        return text

    # Count token frequencies
    from collections import Counter
    counts = Counter(t.lower() for t in tokens)
    total = len(tokens)
    unique = len(counts)

    # If ratio of unique to total is very low, it's noise
    if total > 20 and unique / total < NOISE_UNIQUE_RATIO:
        # Keep only unique tokens (up to 2 occurrences each)
        seen: dict[str, int] = {}
        compressed: list[str] = []
        for token in tokens:
            key = token.lower()
            seen[key] = seen.get(key, 0) + 1
            if seen[key] <= 2:
                compressed.append(token)
        return " ".join(compressed)

    # For non-noisy text, just cap individual token repeats at 3
    seen_counts: dict[str, int] = {}
    result: list[str] = []
    for token in tokens:
        key = token.lower()
        seen_counts[key] = seen_counts.get(key, 0) + 1
        if seen_counts[key] <= 3:
            result.append(token)
    return " ".join(result)
