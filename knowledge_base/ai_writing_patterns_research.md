# AI Writing Patterns & Watermarking Extraction Research

This document synthesizes empirical findings from computational linguistics literature, Wikipedia's "Signs of AI writing" guidelines, Anthropic Claude provenance reports, and multi-vendor watermark removal implementations (`watermarks-remover`, `claude-watermark-remover`, `remove-ai-watermarks`).

---

## 1. Multi-Layer AI Provenance Architecture

Modern AI generation leaves signatures across three distinct layers:

### Layer A: Invisible Unicode & Syntactic Formatting Watermarks
- **Zero-Width Codepoints:** Zero-width spaces (`U+200B`), non-joiners (`U+200C`), joiners (`U+200D`), word joiners (`U+2060`), and BOM (`U+FEFF`).
- **Exotic Space Homoglyphs:** Non-breaking spaces (`U+00A0`), en/em spaces (`U+2002`, `U+2003`), thin/hair spaces (`U+2009`, `U+200A`), narrow no-break spaces (`U+202F`), and ideographic spaces (`U+3000`).
- **Directional Overrides & Isolates:** Bidi embedding and override marks (`U+202A` - `U+202E`, `U+2066` - `U+2069`).
- **Steganography Tag Characters:** Supplementary tags (`U+E0001` - `U+E007F`) and Private Use Area codepoints.
- **Punctuation Artifacts:** Overuse of em dashes (`—`), curly quotation marks (`“” ‘’`), and decorative emojis.

### Layer B: Statistical Token-Sampling Watermarks
- **Mechanism:** Implemented by labs to comply with regulatory mandates (such as the EU AI Act). The generator uses a pseudo-random key to bias token sampling toward a "green-list" of synonyms.
- **Examples:**
  - Google SynthID-Text
  - Anthropic Claude (live on Fable 5.1, Mythos 5.1)
  - Kirchenbauer et al. green-list watermarking
  - Aaronson keyed-Gumbel / EXP watermarking
- **Disruption Method:**
  - Cannot be removed by simply re-saving or tweaking isolated words.
  - Requires a token-level rewriting pass that shifts clause structures, changes sentence lengths, alters conjunctions/connectors, and uses alternate vocabulary.
  - **Crucial Rule:** Claude-generated text should ideally be rewritten by Gemini or GPT, and vice versa, to avoid re-imprinting the same vendor's statistical key.

### Layer C: Container & Document Metadata
- **Word / PowerPoint (.docx, .pptx):** OpenXML properties (`docProps/core.xml`, `docProps/app.xml`) tracking `dc:creator`, `cp:lastModifiedBy`, and company tags.
- **Markdown / HTML:** YAML frontmatter keys (`generator:`, `ai:`, `model:`, `prompt:`), `<meta name="generator">` tags, and JSON-LD metadata.
- **Leaked Model / Search Tokens:** RAG artifacts like `turn0search0`, `oaicite`, `contentReference`, `attributableIndex`, and `cite_turn`.

---

## 2. Stylistic & Linguistic Markers (Wikipedia Signs of AI Writing)

### A. Discourse Connectors & Transitions
AI models rely heavily on formal, signposting transitions to connect ideas:
- *Sentence-initial:* "Additionally", "Furthermore", "Moreover", "Consequently", "Notably".
- *Wrap-ups:* "In conclusion", "Overall", "In summary", "It is worth noting that", "It is important to remember".

### B. High-Density AI Vocabulary (The "Delve" Lexicon)
Empirical frequency studies (Kobak et al., 2025; Juzek & Ward, 2025) identify extreme overrepresentation of:
- `delve`, `tapestry`, `testament`, `beacon`, `pivotal`, `crucial`, `vibrant`, `foster`, `underscore`, `interwoven`, `intertwined`, `paradigm`, `synergy`, `intricate`, `meticulous`, `robust`, `showcase`, `garner`, `boast`, `bolster`, `elevate`, `harness`, `navigate`, `resonate`, `transcend`, `revolutionize`, `unwavering`, `multifaceted`, `plethora`.

### C. Syntactic Tropes
1. **Copula Suppression:** Avoiding simple verbs like "is" or "has" in favor of grand substitutes: "The building serves as a clinic" rather than "The building is a clinic".
2. **Trailing Participial Fluff:** Tacking on participial phrases at sentence ends ("..., underscoring its historical importance", "..., highlighting the ongoing need").
3. **Negative Parallelisms:** Repetitive formulaic contrasts ("Not only X, but also Y", "It is not X, but Y").
4. **Triadic Phrasing:** Consistently clustering descriptions into triplets of adjectives or nouns ("vibrant, rich, and enduring").
5. **Uniform Sentence Lengths (Lack of Burstiness):** Writing sentences that consistently span 18–24 words without dynamic variation.
6. **Robotic Introductions & Outline Conclusions:** Formulaic scene-setters followed by speculative concluding sections ("Challenges and Future Prospects").

---

## 3. Recommended Pipeline for Multi-Format Documents

```
[User Upload: .docx / .md / .txt / .pptx]
                     │
                     ▼
       [Phase A: Document Parsing]
 (Extract clean paragraphs, slide text, or markdown)
                     │
                     ▼
  [Phase B: Layer A Pre-Sanitization]
  (Strip zero-width chars, homoglyphs, leak tokens)
                     │
                     ▼
   [Phase C: Intelligent Chunking Engine]
(Token-aware grouping with context & heading preservation)
                     │
                     ▼
 [Phase D: Cross-Model Rewrite (Layer B)]
 (Apply System Humanizer Prompt + Disruption Template)
                     │
                     ▼
  [Phase E: Layer A Post-Sanitization]
 (Clean any new invisible unicode added by the rewriter)
                     │
                     ▼
     [Phase F: Clean Re-assembly]
 (Inject cleaned text into original document structure,
   stripping author/generator metadata from file props)
                     │
                     ▼
       [User Download Cleaned File]
```
