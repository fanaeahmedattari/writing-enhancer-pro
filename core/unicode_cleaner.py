"""
core/unicode_cleaner.py
Deterministic Layer A cleaner: Strips invisible Unicode codepoints, zero-width characters,
exotic space homoglyphs, bidi directional overrides, and steganographic tag characters.
Based on research from watermarks-remover, claude-watermark-remover, and Unicode standards.
"""

from __future__ import annotations

import unicodedata
from typing import Dict, List, Tuple

# Invisible / Zero-width codepoints commonly used for steganography or AI provenance
STRIP_CODEPOINTS: frozenset[int] = frozenset({
    0x00AD,  # soft hyphen
    0x034F,  # combining grapheme joiner
    0x061C,  # Arabic letter mark
    0x115F,  # Hangul choseong filler
    0x1160,  # Hangul jungseong filler
    0x17B4,  # Khmer vowel inherent AQ
    0x17B5,  # Khmer vowel inherent AA
    0x180B,  # Mongolian free variation selector-1
    0x180C,  # Mongolian free variation selector-2
    0x180D,  # Mongolian free variation selector-3
    0x180E,  # Mongolian vowel separator
    0x200B,  # zero width space
    0x200C,  # zero width non-joiner
    0x200D,  # zero width joiner
    0x200E,  # Left-to-Right Mark (LRM)
    0x200F,  # Right-to-Left Mark (RLM)
    0x202A,  # LRE
    0x202B,  # RLE
    0x202C,  # PDF
    0x202D,  # LRO
    0x202E,  # RLO
    0x2060,  # word joiner
    0x2061,  # function application
    0x2062,  # invisible times
    0x2063,  # invisible separator
    0x2064,  # invisible plus
    0x2066,  # LRI
    0x2067,  # RLI
    0x2068,  # FSI
    0x2069,  # PDI
    0x206A, 0x206B, 0x206C, 0x206D, 0x206E, 0x206F,  # directional inhibit
    0xFEFF,  # Byte Order Mark (BOM) / zero-width no-break space
    0xFE00, 0xFE01, 0xFE02, 0xFE03, 0xFE04, 0xFE05, 0xFE06, 0xFE07,
    0xFE08, 0xFE09, 0xFE0A, 0xFE0B, 0xFE0C, 0xFE0D, 0xFE0E, 0xFE0F, # variation selectors
    0xFFF9, 0xFFFA, 0xFFFB,  # interlinear annotation characters
})

# Exotic and homoglyphic spaces to normalize to standard ASCII space U+0020
SPACE_HOMOGLYPHS: Dict[int, str] = {
    0x00A0: " ",  # no-break space
    0x1680: " ",  # Ogham space mark
    0x2000: " ",  # en quad
    0x2001: " ",  # em quad
    0x2002: " ",  # en space
    0x2003: " ",  # em space
    0x2004: " ",  # three-per-em space
    0x2005: " ",  # four-per-em space
    0x2006: " ",  # six-per-em space
    0x2007: " ",  # figure space
    0x2008: " ",  # punctuation space
    0x2009: " ",  # thin space
    0x200A: " ",  # hair space
    0x202F: " ",  # narrow no-break space
    0x205F: " ",  # medium mathematical space
    0x3000: " ",  # ideographic space
}

# Curly quotes and smart punctuation to normalize
PUNCTUATION_NORMALIZATION: Dict[str, str] = {
    "“": '"',
    "”": '"',
    "‘": "'",
    "’": "'",
    "«": '"',
    "»": '"',
    "…": "...",
}

def is_stego_tag_char(cp: int) -> bool:
    """Checks for Unicode steganography tag characters (U+E0001 to U+E007F)."""
    return 0xE0001 <= cp <= 0xE007F

def is_private_use_area(cp: int) -> bool:
    """Checks for Private Use Area codepoints."""
    return (0xE000 <= cp <= 0xF8FF) or (0xF0000 <= cp <= 0xFFFFD) or (0x100000 <= cp <= 0x10FFFD)

def sanitize_unicode(text: str, normalize_quotes: bool = False) -> Tuple[str, Dict[str, int]]:
    """
    Cleans invisible characters, zero-width spaces, exotic space homoglyphs,
    and returns sanitized text along with a detailed report of removed artifacts.
    """
    if not text:
        return "", {"removed_invisibles": 0, "normalized_spaces": 0, "normalized_quotes": 0}

    out_chars: List[str] = []
    removed_invisibles = 0
    normalized_spaces = 0
    normalized_quotes = 0

    for ch in text:
        cp = ord(ch)
        
        # Check if it should be stripped
        if cp in STRIP_CODEPOINTS or is_stego_tag_char(cp) or is_private_use_area(cp):
            removed_invisibles += 1
            continue
            
        # Check if it is an exotic space
        if cp in SPACE_HOMOGLYPHS:
            out_chars.append(SPACE_HOMOGLYPHS[cp])
            normalized_spaces += 1
            continue
            
        # Optional punctuation normalization
        if normalize_quotes and ch in PUNCTUATION_NORMALIZATION:
            out_chars.append(PUNCTUATION_NORMALIZATION[ch])
            normalized_quotes += 1
            continue
            
        out_chars.append(ch)

    cleaned_text = "".join(out_chars)
    # Apply standard Unicode normalization form C
    cleaned_text = unicodedata.normalize("NFC", cleaned_text)

    stats = {
        "removed_invisibles": removed_invisibles,
        "normalized_spaces": normalized_spaces,
        "normalized_quotes": normalized_quotes
    }
    return cleaned_text, stats
