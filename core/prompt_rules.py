"""
core/prompt_rules.py
Humanization, Academic Flow, Tone & Anti-AI Rewriting Specification and Rule Engine.
Synthesized from:
- Computational Linguistics research & AI watermark analyses (Claude, Google SynthID, OpenAI)
- Wikipedia "Signs of AI writing" guidelines
- Peer-reviewed Academic Publishing standards (PMC4548565, PMC10676260, PMC10676253, Elsevier, MDPI, Purdue OWL)
- University Academic Writing & Tone Guidelines (UQ Pressbooks, Curtin UniSkills, Shaw University)
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# ==============================================================================
# 1. ACADEMIC FLOW, TONE & COHESION DIRECTIVES
# ==============================================================================

ACADEMIC_TONE_AND_WORDING_RULES = """
### ACADEMIC TONE, OBJECTIVITY & LEXICAL ELEVATION
1. **Third-Person Objective Perspective:**
   - Strictly avoid first-person ('I', 'me', 'my', 'we', 'our') and second-person ('you', 'your', 'yours') pronouns unless the text explicitly details personal reflective practice.
   - Present evidence objectively: write 'the evidence indicates' or 'the data demonstrate' instead of 'I think' or 'we believe'.
2. **Expansion of Contractions:**
   - Never use contractions. Always write out the full grammatical form: 'do not' (don't), 'cannot' (can't), 'will not' (won't), 'does not' (doesn't), 'did not' (didn't), 'it is' (it's), 'should not' (shouldn't).
3. **Punctuation & Syntax Decorum:**
   - Eliminate all exclamation marks ('!'). Academic prose is measured and understated.
   - Eliminate rhetorical questions. Transform every query into an authoritative declarative or analytical statement.
4. **Scholarly Lexicon Upgrades:**
   - Replace casual/conversational phrasing with precise scholarly vocabulary:
     * 'show' -> 'demonstrate' / 'indicate'
     * 'look into' -> 'investigate' / 'examine'
     * 'kid' -> 'child'
     * 'get' -> 'obtain' / 'acquire'
     * 'give' -> 'provide' / 'administer'
     * 'big' / 'a lot' -> 'substantial' / 'numerous'
     * 'bad' -> 'adverse' / 'deleterious'
5. **Hedging (Cautious Epistemic Modality):**
   - Soften absolute or dogmatic assertions when discussing non-definitive findings.
   - Use nuanced modal verbs ('may', 'might', 'suggests', 'appears to', 'is likely to') rather than unwarranted absolutes ('proves', 'definitely shows', 'obviously').
6. **Numerals & Acronyms:**
   - Spell out whole numbers zero through nine ('zero', 'one', ... 'nine'); use digits for 10 and above.
   - Never start a sentence with a digit (write 'Seventy-five percent...', not '75%...').
   - Define technical acronyms upon their initial appearance.
"""

FLOW_AND_STITCHING_RULES = """
### CONTINUOUS FLOW & CONTEXT PRESERVATION (THE STITCHING PROTOCOL)
1. **The "Because of This" Causal Backbone:**
   - Link consecutive thoughts using cause-and-effect progression ([Idea X] -> Therefore -> [Idea Y] -> But -> [Idea Z]).
   - Every paragraph must advance the central thesis (the "North Star") without meandering into tangential speculation.
2. **The "Hook and Eye" Paragraph Stitching:**
   - Connect paragraphs organically: pick up a key noun, concept, or outcome from the closing sentence of one paragraph and carry it naturally into the opening sentence of the next.
   - Avoid abrupt topic leaps or disconnected bullet-style paragraphs.
3. **Natural Signposts & Connectors:**
   - Use diverse, organic transitions (e.g., 'In contrast', 'Consequently', 'To illustrate', 'Despite these findings', 'Conversely', 'Specifically').
   - Avoid mechanical, formulaic AI signposts (NEVER start paragraphs with 'Additionally,', 'Furthermore,', 'Moreover,', or 'In conclusion,').
"""

PLAGIARISM_AND_PARAPHRASING_RULES = """
### SCIENTIFIC PARAPHRASING & PLAGIARISM REMOVAL DIRECTIVES (N-GRAM BREAKING)
1. **The "Read and Shield" Cognitive Principle:**
   - Reconstruct the core concepts completely in your own voice rather than performing simple synonym substitution (rogeting).
   - Absorb the factual argument and express it using brand-new sentence architecture.
2. **Syntactic Flipping (Clause & Voice Inversion):**
   - Invert cause-and-effect structures: if the original begins with the cause and ends with the effect, reconstruct the sentence to lead with the effect as the subject.
   - Alternate between active and analytical passive voice appropriately to break source dependency trees.
3. **Part-of-Speech Transformation:**
   - Shift grammatical roles across phrases (e.g., convert verbs into nominalized noun phrases or adjectives into analytical adverbs) to force completely new surrounding syntax.
4. **Sentence Partitioning & Synthesis:**
   - Break monolithic multi-clause sentences (30+ words) into two crisp, declarative statements.
   - Weave fragmented ideas into unified, cohesive thoughts using formal conjunctive transitions.
5. **Strict N-Gram Disruption Mandate:**
   - Never replicate consecutive sequences of 4 or more identical words from the source text (except proper nouns, technical terms, and standard citations).
   - Maintain 100% fidelity for all data points, dates, author citations, and numerical metrics while ensuring zero verbatim phrase matches.
"""


SECTION_SPECIFIC_GUIDELINES: Dict[str, str] = {
    "abstract": """
- **Academic Abstract Protocol (IMRaD & PMC6398294 Standard):**
  * **Sequence (The 5-Part IMRaD Flow):**
    1. Background / Context (1–2 sentences): Introduce the broad scientific topic and state the specific research gap.
    2. Aim or Objective (1 sentence): State explicitly what this specific study investigated or developed.
    3. Methods (2–3 sentences): Summarize the research design, participants/samples, experimental procedures, and analytical tools (without low-level lab trivia).
    4. Results (2–3 sentences): Highlight primary empirical findings, key statistical metrics (p-values, confidence intervals, percentages, effect sizes).
    5. Conclusion / Significance (1–2 sentences): State the overarching take-home message, real-world implications, and broader scientific significance.
  * **Tenses:**
    - Present Tense: For established truths, general background facts, and paper conclusions ("This study demonstrates...").
    - Past Tense: For completed study procedures, experiments, and specific results ("Participants completed...", "The algorithm achieved 94% accuracy").
    - STRICTLY NO Future Tense: The research is completed.
  * **Strict Stand-Alone & Self-Containment Rules:**
    - NEVER cite outside literature or references inside an abstract (e.g., no '[1]', no 'Smith et al. (2020)').
    - NEVER cite figures, tables, or charts from the main text (e.g., no 'Figure 1', no 'Table 2').
    - Avoid nonstandard abbreviations unless defined at first use.
  * **Style & Perspective:** Objective, third-person perspective, active voice preferred, concise and direct.
  * **Keywords:** Conclude the abstract with 3–6 relevant indexing keywords (preferably MeSH compatible): 'Keywords: term1, term2, term3'.
""",
    "introduction": """
- **Introduction Protocol:**
  * **Tenses:** Present tense for general truths, background facts, and current research problems. Simple past tense when referring to specific previous studies or historical findings.
  * **Sentence Architecture:** Complex and compound-complex sentences linking broader context with the specific research gap.
  * **Flow:** Move from general landscape -> specific problem/gap -> clear declaration of the study's objective or hypothesis.
""",
    "literature_review": """
- **Literature Review Protocol:**
  * **Tenses:** Present tense or present perfect for established theories, frameworks, or ongoing scientific consensus. Simple past tense for specific historical author experiments ("Smith (2021) demonstrated...").
  * **Sentence Architecture:** Periodic and subordinate clauses to juxtapose contrasting viewpoints and synthesize thematic clusters.
  * **Flow:** Thematic or methodological groupings, never isolated serial summaries.
""",
    "methodology": """
- **Methodology Protocol:**
  * **Tenses:** Simple past tense for all procedures, experiments, participant recruitment, and tools used. Present tense only when referencing static elements (figures, formulas, tables: "Table 1 lists...", "Equation 2 defines...").
  * **Sentence Architecture:** Direct declarative sentences in active or past-passive voice detailing exact chronological execution.
  * **Flow:** Precise, reproducible steps without subjective commentary or qualitative fluff.
""",
    "results": """
- **Results Protocol:**
  * **Tenses:** Simple past tense to report experimental data, statistical metrics, and observed outcomes. Present tense when pointing to charts or tables ("Figure 2 shows...").
  * **Sentence Architecture:** Concise descriptive sentences backed by exact numerical and statistical data.
  * **Fidelity:** Strict separation of data from interpretation. Do NOT interpret or theorize in the Results section.
""",
    "discussion": """
- **Discussion Protocol:**
  * **Tenses:** Present tense to interpret findings, state implications, and evaluate theoretical significance. Simple past tense to briefly recap key observed metrics. Modals ('may', 'suggests') for epistemic hedging.
  * **Sentence Architecture:** Evaluative complex sentences synthesizing findings and connecting them back to the primary hypothesis.
  * **Flow:** Direct declaration of major findings in paragraph 1 -> relation and comparison to previous literature -> alternative explanations and rival interpretations -> thorough, honest examination of study limitations.
""",
    "conclusion": """
- **Academic Conclusion Protocol (Grammarly & MDPI Framework):**
  * **The 4-Step Structural Flow:**
    1. Restate the Central Thesis (1–2 sentences): Paraphrase the central claim, research hypothesis, or overarching objective using fresh, elevated vocabulary—NEVER copy verbatim from the introduction or abstract.
    2. Synthesize Key Contributions (2–3 sentences): Interconnect the core empirical findings to demonstrate how the evidence unites into a cohesive narrative, rather than producing a disjointed serial list.
    3. Address the 'So What?' (Broader Context & Significance) (2–3 sentences): Explicitly articulate why these outcomes matter to the discipline. Situate the study within the broader academic discourse and explain its theoretical, clinical, or technological implications.
    4. Provide a Forward-Looking Closing Thought (1–2 sentences): Formulate a compelling final thought—propose actionable future research trajectories tied directly to study constraints, recommend methodological extensions, or deliver an authoritative concluding takeaway.
  * **Critical Scientific Constraints (MDPI Standards):**
    - Summarize, Don't Repeat: Synthesize and legitimize paper arguments rather than reiterating previously stated paragraphs.
    - Zero New Evidence or Untested Theories: Strictly NEVER introduce novel data, unanalyzed literature citations, or speculative theories not evaluated in the main body.
    - The 10% Length Guideline: Keep the conclusion concise and balanced (typically ~5–10% of total document length).
  * **Banned Conclusion Clichés:**
    - NEVER start paragraphs with 'In conclusion,', 'In summary,', 'To sum up,', 'All in all,', 'Finally,', 'In a nutshell,', or 'It is concluded that'. Begin directly with the substantive thematic assertion.
  * **Tenses & Modality:**
    - Present Tense: For permanent claims, overarching implications, and enduring conclusions ("This architecture establishes...").
    - Past / Present Perfect: To briefly recap completed empirical demonstrations ("The experimental evaluation demonstrated...").
    - Modals & Epistemic Hedging: Use 'warrants', 'suggests', 'indicates', 'should explore' when recommending future research vectors.
""",
    "general": """
- **Standard Academic Tone:**
  * Formal, objective, analytical, and unambiguous.
  * Strict third-person perspective; zero colloquialisms and zero contractions.
  * Active voice for clarity and speed; selective passive voice when the process/object is the focal point.
"""
}

WORK_MODE_GUIDELINES: Dict[str, str] = {
    "journal": """- **Work Mode: Peer-Reviewed Journal Article (IMRaD Standard):**
  * Prioritize empirical density, crisp causal claims, and concise, high-impact phrasing.
  * Explicitly separate data assertions from theoretical interpretations.
  * Adhere strictly to top-tier journal publishing standards (Nature, IEEE, Elsevier, Springer).""",

    "thesis": """- **Work Mode: Master's / PhD Dissertation & Thesis:**
  * Adopt an exhaustive, deeply grounded scholarly register.
  * Provide rigorous theoretical justification, comprehensive contextualization, and elaborate methodological clarity.
  * Maintain formal chapter-level academic depth with clear signposting between subsections.""",

    "review_paper": """- **Work Mode: Systematic / Comprehensive Literature Review:**
  * Prioritize high-order thematic synthesis over serial author summaries.
  * Juxtapose conflicting paradigms, synthesize consensus findings, and identify unexplored empirical gaps.
  * Group findings by methodology, theme, or chronological evolution.""",

    "conference": """- **Work Mode: Conference Proceedings (IEEE / ACM / Springer):**
  * Maximum technical conciseness and high informational density.
  * Lead with concrete computational, empirical, or algorithmic contributions without unnecessary introductory fluff.""",

    "grant_proposal": """- **Work Mode: Research Grant Proposal & Scientific Pitch:**
  * Adopt a compelling, persuasive yet rigorous scholarly tone.
  * Clearly articulate problem significance, methodology feasibility, risk mitigation, and transformative societal/scientific impact.""",

    "essay": """- **Work Mode: Coursework, Academic Essay & Term Paper:**
  * Clear pedagogical argumentation structured around a distinct central thesis statement.
  * Logical paragraph topic sentences supported by balanced critical evidence and coherent transitions.""",

    "assignment": """- **Work Mode: University Coursework & Graded Student Assignment:**
  * Specifically calibrated for undergraduate and graduate coursework, problem sets, case studies, and university assignments.
  * Emphasize direct, authoritative answers to the assignment prompts with rigorous theoretical grounding and clear conceptual definitions.
  * Organize responses with logical, structured subheadings; eliminate conversational filler and superficial fluff.
  * Maintain an authentic scholarly student voice demonstrating deep subject mastery and accurate academic citations."""
}

ENGLISH_TONE_GUIDELINES: Dict[str, str] = {
    "academic": """- **English Tone: Academic Rigorous (Global Scholarly):**
  * Uncompromising scholarly objectivity, high-register academic vocabulary, and disciplined third-person voice.
  * Practice systematic epistemic hedging ('the data indicate', 'evidence suggests' instead of absolute claims).""",

    "professional": """- **English Tone: Professional & Scientific Executive:**
  * Crisp, direct, authoritative, and jargon-controlled.
  * Active voice prioritized for maximum executive clarity, eliminating academic convolution while retaining intellectual gravitas.""",

    "native_us": """- **English Tone: Native US Academic (American Standard - APA / Chicago):**
  * Follow standard American spelling conventions: '-ize' (analyze, synthesize, prioritize), '-or' (behavior, labor, vigor), '-er' (center, meter), 'program', 'judgment'.
  * Adhere to APA/Chicago punctuation (serial Oxford comma, double quotation marks for direct quotes with punctuation inside).
  * Natural native American scholarly rhythm with smooth idiomatic transitions.""",

    "native_uk": """- **English Tone: Native UK / Oxford Academic (British Standard):**
  * Follow standard British/Commonwealth spelling conventions: '-ise' or '-ize' (Oxford standard), '-our' (behaviour, labour, vigour), '-re' (centre, metre), 'programme', 'judgement'.
  * Adhere to British punctuation (single quotation marks for quotes with punctuation outside unless part of quoted matter).
  * Measured, understated British academic cadence with formal syntactic restraint.""",

    "indo_pak": """- **English Tone: Indo-Pak / Global Scholarly English (Regional Idiom Normalizer):**
  * Specially calibrated for South Asian and Pakistani scholars submitting to international indexed journals and HEC-recognized dissertations.
  * CRITICAL: Detect and automatically convert characteristic South Asian / Pakistani English idioms into internationally standard native scholarly equivalents:
    - Replace 'revert back' -> 'respond', 'reply', or 'return'.
    - Replace 'do the needful' -> 'take the requisite measures' or 'implement the appropriate action'.
    - Replace 'passed out of university/college' -> 'graduated' or 'completed their degree'.
    - Replace 'inculcate' -> 'foster', 'develop', or 'instill'.
    - Replace 'intimate the committee/department' -> 'notify', 'inform', or 'apprise'.
    - Replace 'discuss about' -> 'discuss' or 'examine'.
    - Replace 'prepone' -> 'advance', 'reschedule to an earlier date', or 'bring forward'.
    - Replace 'batchmates' / 'colleagues' -> 'peers', 'cohort', or 'fellow researchers'.
    - Replace 'up to the mark' -> 'satisfying established benchmarks' or 'meeting quality criteria'.
    - Replace 'out of station' -> 'away', 'off-site', or 'travelling'.
  * Preserve the polite, dignified, and formal scholarly depth traditional in regional academia while guaranteeing 100% native international reception."""
}

STRICT_MODE_DIRECTIVE: str = """- **STRICT ACADEMIC ENFORCEMENT & ANTI-DETECTION MANDATE (Turnitin, ZeroGPT, GPTZero, Copyleaks, Winston AI):**
  * ABSOLUTE BAN ON EM DASHES ('—') AND DOUBLE HYPHENS ('--'): Modern AI classifiers (Turnitin, ZeroGPT, GPTZero, Winston AI) heavily penalize em dashes as machine parentheticals. NEVER generate em dashes ('—') or double hyphens ('--'). Use commas, parentheses, or separate standalone sentences.
  * RADICAL BURSTINESS (DEFEATING STATISTICAL DETECTORS): Deliberately vary sentence length across every paragraph. Pair short punchy declarative sentences (4 to 8 words) with longer, multi-clause analytical sentences (24 to 34 words). Never write consecutive sentences of uniform length.
  * HIGH PERPLEXITY & ANTI-PREDICTABILITY: Prohibit predictable AI tokens ('plays a pivotal role', 'testament to', 'delve into', 'vibrant tapestry', 'multifaceted', 'enduring shift', 'current societal dynamics', 'furthermore', 'moreover'). Use concrete, active human academic phrasing.
  * TURNITIN & iTHENTICATE N-GRAM TRANSLOCATION: Break consecutive 8-to-16-word shingles. Invert dependent and independent clauses, shift between active and selective passive voice, and restructure compounds into contrasting arguments.
  * Absolute Contraction Prohibition: ZERO contractions permitted under any circumstance ('don't' -> 'do not', 'can't' -> 'cannot', 'it's' -> 'it is', 'they've' -> 'they have').
  * Strict 100% Third-Person Objectivity: Absolute ban on first-person ('I', 'me', 'my', 'we', 'our', 'us') and second-person ('you', 'your').
  * Zero Conversational & Colloquial Vocabulary: Strictly eliminate casual words ('a lot', 'big', 'get', 'kid', 'stuff', 'thing', 'pretty good', 'nowadays').
  * Zero Rhetorical Questions & Exclamation Marks: Every sentence must be an informative declarative or analytical statement.
  * Active Epistemic Hedging: Avoid dogmatic absolutes; temper claims using scholarly qualifiers ('the data indicate', 'evidence suggests', 'probable indicator')."""


# ==============================================================================
# 2. MASTER SYSTEM PROMPT
# ==============================================================================

SYSTEM_HUMANIZER_PROMPT = f"""You are an elite academic and professional editorial humanizer, computational linguist, and master stylist. Your sole task is to rewrite the provided text to completely eliminate all mechanical, statistical, and stylistic markers of AI-generated text while preserving 100% of the factual accuracy, technical context, proper nouns, and core intent.

Adhere rigorously to the following specification:

## SECTION 1: STRICTLY BANNED VOCABULARY & PHRASES (DO NOT USE)

### A. Discourse Transitions & Signposts
- NEVER use: 'additionally' (especially sentence-initial), 'furthermore', 'moreover', 'consequently', 'notably', 'in conclusion', 'in summary', 'overall', 'importantly', 'as a result', 'in today\'s evolving landscape', 'in today\'s society', 'it is worth noting', 'it is important to remember', 'it is crucial to consider'.
- Action: Omit entirely, integrate directly into the clause, or use natural simple connectors ('and', 'but', 'so', 'also').

### B. Overused AI Buzzwords & High-Probability Lexicon
- NEVER use: 'delve' / 'delving', 'tapestry', 'testament', 'beacon', 'pivotal', 'crucial', 'foster' / 'fostering', 'underscore' / 'underscores', 'vibrant', 'interwoven', 'intertwined', 'paradigm', 'synergy', 'synergies', 'rich', 'enduring', 'intricate' / 'intricacies', 'interplay', 'landscape' (abstract), 'realm', 'meticulous' / 'meticulously', 'robust', 'showcase' / 'showcases', 'highlight' (as verb), 'garner', 'boast' / 'boasts', 'bolster', 'elevate', 'embrace', 'harness', 'navigate' / 'navigating', 'resonate' / 'resonates', 'transcend', 'revolutionize', 'unwavering', 'multifaceted', 'plethora', 'myriad', 'shed light on'.
- Action: Use direct, domain-specific verbs and plain adjectives (e.g., 'examine' or 'study' instead of 'delve into'; 'shows' instead of 'serves as a testament to'; 'has' instead of 'boasts'; 'used' instead of 'utilized').

### C. Robotic Qualifiers & Attributions
- NEVER use: 'serves as', 'stands as', 'functions as', 'marks a pivotal moment', 'represents a significant shift', 'underscores the importance of', 'reflects broader trends', 'setting the stage for', 'align with', 'in connection with', 'associated with', 'independent coverage', 'a testament to human ingenuity'.
- Action: Use standard copulatives ('is', 'are', 'was') or state relationships directly without throat-clearing.

## SECTION 2: SYNTACTIC & STRUCTURAL PATTERNS TO ELIMINATE

1. **Restore Natural Copulatives (Stop Copula Suppression):** Replace pretentious verbs like 'serves as', 'stands as', 'features', 'boasts' with simple, direct forms ('is', 'are', 'was', 'has').
2. **Cut Superficial Trailing Participles (-ing Tails):** Eliminate tacked-on participial phrases designed to manufacture unearned profundity (e.g., cut '...underscoring its historical importance', '...ensuring a vibrant future', '...contributing to the ongoing dialogue'). End the sentence on the concrete action or fact.
3. **Eliminate Negative Parallelisms:** Remove artificial debunking contrasts such as 'Not only X, but also Y', 'It is not X, but Y', 'Y rather than X'. State claims directly.
4. **Break Triadic Phrasing (Rule of Three):** Avoid rhythmic triplets of adjectives, nouns, or verbs (e.g., 'vibrant, rich, and enduring'). Use one or two precise terms instead.
5. **Abolish Paragraph Symmetry & Inject Burstiness:** Vary sentence length dynamically. Combine punchy 3-6 word sentences with longer, complex compound sentences (20-30 words). Never write consecutive paragraphs of identical line count.
6. **No AI Cliché Hooks or Outline Summaries:** Never begin with sweeping generic scene-setters ('In today\'s fast-paced world...', 'In today\'s society...') and NEVER append mandatory wrap-up summaries ('In conclusion', 'Future Prospects', 'Challenges and Legacy', 'Looking ahead'). Cut straight to substance and stop when the information ends.
7. **Prose Over Bold Lists:** Transform bullet lists with bold inline titles (`* **Key:** Description`) into natural flowing paragraphs unless technical specifications explicitly require a list.
8. **Absolute Ban on Em Dashes & Double Hyphens:** NEVER use em dashes ('—') or double hyphens ('--') for parenthetical thoughts. Turnitin, ZeroGPT, GPTZero, Copyleaks, and Winston AI heavily penalize parenthetical em dashes as machine signatures. Use standard commas, parentheses, or separate standalone sentences.
9. **Turnitin & iThenticate N-Gram Decoupling:** Defeat consecutive 8-to-16-word shingle matching. Transmute syntactic order by inverting dependent and independent clauses, shifting voice (active to passive or vice versa), and decomposing compound sentences into direct, distinct claims.
10. **Radical Length Asymmetry (Burstiness Injection):** Avoid monotone rhythms where all sentences have 15-22 words. Force contrast: follow an 8-word statement with a 28-word multi-clause analysis, then a 14-word synthesis. ZeroGPT and Sapling AI rely on rhythmic variance to authenticate human origin.

{ACADEMIC_TONE_AND_WORDING_RULES}

{FLOW_AND_STITCHING_RULES}

## SECTION 3: HUMAN STYLE & ACADEMIC CADENCE INJECTORS

1. **Rhythmic Burstiness:** Mix staccato statements with flowing compound structures to replicate authentic human cognition.
2. **Active Voice & Subject-Verb Pacing:** Keep subjects and verbs close together. Prefer 'The team found' over 'A discovery was made by the team'.
3. **Conversational Directness & Asymmetry:** Allow natural pacing and occasional plain understatement. Avoid forced optimism, neat philosophical bows, or pseudo-poetic conclusions.
4. **Precision & Formality:** Avoid informal contractions ('do not' instead of 'don\'t'). Maintain academic objectivity.

## SECTION 4: CONTEXT & FIDELITY MANDATE (100% PRESERVATION)

1. Retain 100% of original factual claims, logical arguments, numbers, measurements, dates, and proper nouns verbatim.
2. Preserve all specialized domain jargon, technical identifiers, formulas, citations, and reference markers intact.
3. Do NOT summarize, condense, or hallucinate new facts. Output ONLY the rewritten text.
"""

# ==============================================================================
# 3. REWRITE TEMPLATES WITH SECTION AWARENESS
# ==============================================================================

def build_length_constraint_directive(
    text_chunk: str,
    length_mode: str = "preserve",
    target_min: Optional[int] = None,
    target_max: Optional[int] = None,
) -> str:
    """
    Constructs an explicit, mathematically bounded length and word count directive.
    Prevents the typical 20-30% shrinkage that occurs when eliminating AI fluff,
    or enforces exact targets (e.g. 250-300 words for journal abstracts).
    """
    orig_words = len(text_chunk.split())

    if length_mode == "target_range" and target_min is not None and target_max is not None:
        return f"""- **Strict Target Word Count Window ({target_min}–{target_max} Words Mandate):**
  * Original input contains **{orig_words} words**.
  * The final output MUST strictly fall within **{target_min} to {target_max} words**.
  * If the input is below {target_min} words, do NOT insert fictional data; instead, expand with rigorous methodological explanation, scientific context, and scholarly analytical depth.
  * If the input exceeds {target_max} words, synthesize concisely to eliminate verbosity while retaining all essential findings.
  * Self-Audit: Verify that the rewritten word count is strictly within the [{target_min}, {target_max}] word interval."""

    elif length_mode == "concise":
        target = max(30, int(orig_words * 0.75))
        return f"""- **Tighten & Condense Directive (~{target} words, -25%):**
  * Original input contains **{orig_words} words**.
  * Condense the prose by ~25% (aim for ~{target} words), eliminating wordiness and repetitive phrasing while retaining 100% of factual assertions."""

    elif length_mode == "elaborate":
        target = int(orig_words * 1.25)
        return f"""- **Elaborate & Deepen Directive (~{target} words, +25%):**
  * Original input contains **{orig_words} words**.
  * Expand the prose by ~25% (aim for ~{target} words) by providing richer scholarly explanation, contextual nuance, and precise academic elaboration."""

    else:  # "preserve" (default)
        min_bound = max(20, orig_words - 15)
        max_bound = orig_words + 25
        return f"""- **Word Count & Length Preservation Directive:**
  * Original input contains **{orig_words} words**.
  * Maintain approximately the same word count (±5–10% of original: target range **{min_bound} to {max_bound} words**).
  * CRITICAL: Do NOT drastically shrink or truncate the text. When removing AI filler and empty signposts, replace them with substantive scholarly depth, concrete specifics, and precise explanatory vocabulary so the final word count remains balanced."""


def get_user_rewrite_prompt(
    text_chunk: str,
    section_type: str = "general",
    aggressive: bool = False,
    options: Optional[Dict[str, bool]] = None,
    prev_context_tail: str = "",
    length_mode: str = "preserve",
    target_min: Optional[int] = None,
    target_max: Optional[int] = None,
    work_mode: str = "journal",
    english_tone: str = "academic",
    strict_mode: bool = False,
) -> str:
    """
    Generates tailored user rewrite prompt with section-specific grammatical directives,
    target word count / length constraints, user-selected transformation options,
    work mode (journal, thesis, etc.), English tone (US, UK, Indo-Pak, etc.),
    strict academic mode, and Hook-and-Eye cross-chunk stitching context.
    """
    sec_guidelines = SECTION_SPECIFIC_GUIDELINES.get(section_type.lower(), SECTION_SPECIFIC_GUIDELINES["general"])
    
    # Default all options to True if not provided
    opts = {
        "humanize": True,
        "academic_tone": True,
        "flow_stitching": True,
        "sentence_structure": True,
        "tables_and_figures": True,
        "plagiarism_remover": True,
    }
    if options:
        opts.update(options)

    directives: List[str] = []

    # Dynamic word count constraint (first directive to establish length boundary)
    directives.append(build_length_constraint_directive(text_chunk, length_mode, target_min, target_max))

    # Work Mode guideline
    if work_mode and work_mode.lower() in WORK_MODE_GUIDELINES:
        directives.append(WORK_MODE_GUIDELINES[work_mode.lower()])

    # English Tone guideline
    if english_tone and english_tone.lower() in ENGLISH_TONE_GUIDELINES:
        directives.append(ENGLISH_TONE_GUIDELINES[english_tone.lower()])

    # Strict Mode directive
    if strict_mode:
        directives.append(STRICT_MODE_DIRECTIVE)

    if opts.get("humanize", True):
        directives.append("- **Anti-AI Writing Rules:** Eliminate all overused AI buzzwords ('delve', 'tapestry', 'testament', 'pivotal'), remove formulaic transitions ('furthermore', 'moreover', 'additionally'), break triadic phrasing, and abolish robotic qualifiers ('serves as', 'stands as').")

    if opts.get("academic_tone", True):
        directives.append("- **Academic Tone & Lexicon:** Strictly enforce third-person objective perspective (zero 'I', 'we', 'you'). Expand all contractions completely ('do not', 'cannot', 'it is'). Upgrade colloquial verbs to precise scholarly terms, eliminate exclamation marks and rhetorical questions, and practice cautious epistemic hedging ('may indicate', 'appears to').")

    if opts.get("flow_stitching", True):
        directives.append("- **Continuous Flow & Stitching:** Apply the 'Because of This' causal backbone so thoughts advance logically. Use organic, diverse transition phrases rather than repetitive mechanical signposts.")

    if opts.get("sentence_structure", True):
        directives.append("- **Dynamic Sentence Architecture:** Inject high burstiness by varying sentence lengths (staccato 4-7 word statements paired with rich compound sentences). Eliminate negative parallelisms ('not only X but also Y') and cut superficial trailing '-ing' participles.")

    if opts.get("tables_and_figures", True):
        directives.append(
            "- **Scientific Figures, Visual Interpretation & Data Fidelity:**\n"
            "  * Factual Invariance: When describing what a figure, diagram, or chart illustrates, preserve 100% of the underlying scientific reality. Never fabricate unobserved phenomena or alter biochemical/physical claims (e.g. do not turn hydrophobic contacts into covalent bonds).\n"
            "  * Panel & Callout Anchors: Retain exact Figure/Table numbering (e.g. 'Figure 3.1', 'Table 2.1') and panel designations (e.g. '(A)', '(B)', '(C)', '(D)'). Never swap, omit, or misattribute panels.\n"
            "  * Numerical & Metric Veracity: Any distance (e.g., 2.85 Å), binding energy (e.g., -8.6 kJ/mol), or statistical threshold associated with a figure must be preserved verbatim."
        )

    if opts.get("plagiarism_remover", True):
        directives.append("- **Scientific Paraphrasing & N-Gram Disruption:** Apply the 'Read and Shield' protocol to rebuild sentences completely from scratch. Invert syntactic clauses, shift parts of speech, and strictly break all 4+ word consecutive matching sequences from the source text while preserving 100% of factual citations and numerical data.")

    # Hook and Eye cross-chunk stitching context
    context_directive = ""
    if prev_context_tail and opts.get("flow_stitching", True):
        clean_tail = prev_context_tail.strip().replace("\n", " ")
        context_directive = f"""
### CROSS-CHUNK STITCHING CONTEXT:
The preceding section concluded with the following thought:
"{clean_tail}"
Ensure the opening sentence of this new chunk links naturally to this thought using Hook-and-Eye paragraph stitching.
"""

    intensity_header = "Aggressive Anti-AI & Watermark Disruption Mode" if aggressive else "Balanced Academic & Stylistic Enhancement Mode"

    directives_str = "\n".join(directives)

    return f"""Please rewrite the following text block in {intensity_header}.

{sec_guidelines}

### ACTIVE DIRECTIVES:
{directives_str}
{context_directive}
### INVIOLABLE FIDELITY MANDATE:
1. Preserve 100% of facts, experimental data, numbers, dates, proper nouns, and citations verbatim.
2. Do NOT summarize, truncate, or hallucinate content.
3. Output ONLY the rewritten text without conversational preamble ("Here is the rewritten text:"), markdown code wrappers, or post-explanations.

--- ORIGINAL TEXT BLOCK START ---
{text_chunk}
--- ORIGINAL TEXT BLOCK END ---
"""

build_chunk_user_prompt = get_user_rewrite_prompt


USER_REWRITE_TEMPLATE = """Please humanize the following text block. Strictly adhere to all anti-AI vocabulary exclusions, structural directives, copulative restorations, and context preservation mandates.

--- ORIGINAL TEXT BLOCK START ---
{text_chunk}
--- ORIGINAL TEXT BLOCK END ---

Output ONLY the rewritten text without meta-commentary, introductory notes (such as "Here is the humanized text:"), quotation wrappers, or post-explanations.
"""

# ==============================================================================
# 4. AUDIT LEXICON & PATTERN COLLECTIONS
# ==============================================================================

BANNED_TRANSITIONS: List[str] = [
    "furthermore", "moreover", "in conclusion", "in summary", "overall",
    "additionally", "notably", "consequently", "importantly", "as a result",
    "it is worth noting", "it is important to note", "it is crucial to consider",
    "in today's evolving landscape", "in today's fast-paced world", "in today's society", "to sum up"
]

FILLER_BUZZWORDS: List[str] = [
    "delve", "delves", "delving", "tapestry", "testament", "beacon",
    "pivotal", "crucial", "foster", "fostering", "fosters", "underscore",
    "underscores", "underscoring", "vibrant", "interwoven", "intertwined",
    "paradigm", "synergy", "synergies", "enduring", "intricate", "intricacies",
    "interplay", "meticulous", "meticulously", "robust", "showcase", "showcases",
    "showcasing", "garner", "garnered", "boast", "boasts", "bolster", "bolstered",
    "elevate", "elevates", "embrace", "embraces", "harness", "harnessing",
    "navigate", "navigating", "resonate", "resonates", "transcend", "transcends",
    "revolutionize", "unwavering", "multifaceted", "plethora", "myriad"
]

ROBOTIC_QUALIFIERS: List[str] = [
    "serves as", "stands as", "functions as", "marks a pivotal moment",
    "represents a significant shift", "underscores the importance of",
    "reflects broader trends", "setting the stage for", "a testament to",
    "testament to human ingenuity"
]

LLM_LEAK_PATTERNS: List[re.Pattern] = [
    re.compile(r"turn\d+search\d+", re.I),
    # contentReference must be matched BEFORE oaicite to prevent partial consumption
    re.compile(r"contentReference\[[^\]\n]+\](?:\{[^\}\n]+\})?", re.I),
    re.compile(r"attributableIndex\[[^\]\n]+\](?:\{[^\}\n]+\})?", re.I),
    re.compile(r"oaicite:?\{?[^\}\n]+\}?", re.I),
    re.compile(r"cite_turn\d+", re.I),
    re.compile(r"\[citation needed\]", re.I),
    re.compile(r"As an AI language model", re.I),
    re.compile(r"I cannot fulfill this request", re.I),
    re.compile(r"Here is a revised version:", re.I),
    re.compile(r"Certainly! Here is", re.I),
]

NEGATIVE_PARALLELISM_RE = re.compile(
    r"\b(not only\b.*?\bbut also|not just\b.*?\bbut also|it is not\b.*?\bbut\b|rather than\b)",
    re.I
)

TRAILING_PARTICIPLE_RE = re.compile(
    r",\s+(underscoring|highlighting|ensuring|contributing|reflecting|fostering|marking|serving as)\b",
    re.I
)

CONTRACTION_RE = re.compile(r"\b(don't|can't|won't|it's|doesn't|didn't|hasn't|haven't|shouldn't|wouldn't|couldn't|I'm|we're|you're|they're)\b", re.I)

FIRST_SECOND_PERSON_RE = re.compile(r"\b(I|me|my|mine|myself|you|your|yours|yourself|we|us|our|ours)\b", re.I)

SUBJECTIVE_OPINION_RE = re.compile(r"\b(I think|I feel|I believe|in my opinion|in our opinion)\b", re.I)

# ==============================================================================
# 5. TEXT AUDITOR & READABILITY SCORER
# ==============================================================================

def analyze_ai_patterns(text: str) -> Dict[str, Any]:
    """
    Inspects input text and returns a granular analysis of detected AI writing signals,
    tone infractions (contractions, personal pronouns, exclamation marks), and estimated AI score.
    """
    if not text:
        return {
            "score": 0.0,
            "level": "Clean",
            "findings_count": 0,
            "banned_transitions": [],
            "filler_words": [],
            "robotic_qualifiers": [],
            "leak_tokens": [],
            "negative_parallelisms": 0,
            "trailing_participles": 0,
            "em_dashes_count": 0,
            "contractions_count": 0,
            "first_second_person_count": 0,
            "exclamations_count": 0,
            "rhetorical_questions_count": 0,
            "subjective_opinions_count": 0,
            "sentence_count": 0,
            "avg_sentence_length": 0.0
        }

    lower_text = text.lower()
    
    # 1. Check banned transitions
    found_transitions = [w for w in BANNED_TRANSITIONS if re.search(r"\b" + re.escape(w) + r"\b", lower_text)]
    
    # 2. Check filler words
    found_fillers = [w for w in FILLER_BUZZWORDS if re.search(r"\b" + re.escape(w) + r"\b", lower_text)]
    
    # 3. Check robotic qualifiers
    found_qualifiers = [w for w in ROBOTIC_QUALIFIERS if w in lower_text]
    
    # 4. Check LLM leaks
    found_leaks = []
    for pattern in LLM_LEAK_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            found_leaks.extend(matches)
            
    # 5. Syntactic & Tone checks
    neg_parallel_count = len(NEGATIVE_PARALLELISM_RE.findall(text))
    participle_count = len(TRAILING_PARTICIPLE_RE.findall(text))
    em_dash_count = text.count("—") + text.count("--")
    contraction_count = len(CONTRACTION_RE.findall(text))
    first_second_person_count = len(FIRST_SECOND_PERSON_RE.findall(text))
    exclamations_count = text.count("!")
    rhetorical_questions_count = text.count("?")
    subjective_count = len(SUBJECTIVE_OPINION_RE.findall(text))
    
    # Sentence stats
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_count = len(sentences)
    words = text.split()
    word_count = len(words) if words else 1
    avg_sentence_len = round(word_count / sentence_count, 1) if sentence_count > 0 else 0.0
    
    # Heuristic scoring (0 to 100)
    total_triggers = (
        len(found_transitions) * 3
        + len(found_fillers) * 2
        + len(found_qualifiers) * 3
        + len(found_leaks) * 10
        + neg_parallel_count * 2
        + participle_count * 2
        + (1 if em_dash_count > 3 else 0)
        + contraction_count * 2
        + subjective_count * 4
    )
    
    # Density score scaled per 100 words
    density = (total_triggers / (word_count / 100.0)) if word_count >= 50 else total_triggers * 2.0
    score = min(100.0, round(density * 12.0, 1))
    
    if score >= 60.0 or len(found_leaks) > 0:
        level = "High AI Probability"
    elif score >= 25.0:
        level = "Moderate AI Signals"
    elif score > 0:
        level = "Low AI Markers"
    else:
        level = "Clean / Human-like"

    return {
        "score": score,
        "level": level,
        "total_triggers": total_triggers,
        "word_count": word_count,
        "sentence_count": sentence_count,
        "avg_sentence_length": avg_sentence_len,
        "banned_transitions": found_transitions,
        "filler_words": found_fillers,
        "robotic_qualifiers": found_qualifiers,
        "leak_tokens": found_leaks,
        "negative_parallelisms": neg_parallel_count,
        "trailing_participles": participle_count,
        "em_dashes_count": em_dash_count,
        "contractions_count": contraction_count,
        "first_second_person_count": first_second_person_count,
        "exclamations_count": exclamations_count,
        "rhetorical_questions_count": rhetorical_questions_count,
        "subjective_opinions_count": subjective_count
    }


def calculate_ngram_similarity(
    original_text: str,
    rewritten_text: str,
    n: int = 4
) -> Dict[str, Any]:
    """
    Computes N-Gram similarity and verbatim phrase matching between original and rewritten text.
    Used for scientific plagiarism risk assessment and verifying that N-gram sequences
    have been effectively broken (defeating 3-7 word consecutive string matching).
    """
    if not original_text or not rewritten_text:
        return {
            "n": n,
            "original_ngrams_count": 0,
            "rewritten_ngrams_count": 0,
            "matching_ngrams_count": 0,
            "overlap_percentage": 0.0,
            "originality_score": 100.0,
            "risk_level": "Clean / 100% Original",
            "matching_sequences": [],
            "three_gram_overlap": 0.0,
            "five_gram_overlap": 0.0,
        }

    def get_tokens(t: str) -> List[str]:
        return re.findall(r"\b[a-zA-Z0-9_-]+\b", t.lower())

    orig_tokens = get_tokens(original_text)
    reph_tokens = get_tokens(rewritten_text)

    def generate_ngrams(tokens: List[str], size: int) -> List[tuple[str, ...]]:
        return [tuple(tokens[i:i + size]) for i in range(len(tokens) - size + 1)]

    # Compute target N-grams
    orig_ngrams = generate_ngrams(orig_tokens, n)
    reph_ngrams = generate_ngrams(reph_tokens, n)

    orig_set = set(orig_ngrams)
    reph_set = set(reph_ngrams)

    common_ngrams = orig_set.intersection(reph_set)
    overlap_pct = round((len(common_ngrams) / max(1, len(orig_set))) * 100.0, 1)
    originality = round(max(0.0, 100.0 - overlap_pct), 1)

    # Also compute 3-gram and 5-gram for multi-depth analysis
    orig_3 = set(generate_ngrams(orig_tokens, 3))
    reph_3 = set(generate_ngrams(reph_tokens, 3))
    overlap_3 = round((len(orig_3.intersection(reph_3)) / max(1, len(orig_3))) * 100.0, 1)

    orig_5 = set(generate_ngrams(orig_tokens, 5))
    reph_5 = set(generate_ngrams(reph_tokens, 5))
    overlap_5 = round((len(orig_5.intersection(reph_5)) / max(1, len(orig_5))) * 100.0, 1)

    # Format sample matching sequences
    matching_phrases = [" ".join(gram) for gram in list(common_ngrams)[:15]]

    if overlap_pct >= 25.0 or overlap_5 > 15.0:
        risk = "High Similarity / Verbatim Matching"
    elif overlap_pct >= 10.0 or overlap_5 > 5.0:
        risk = "Moderate Risk / Partial Overlap"
    elif overlap_pct > 0.0:
        risk = "Low Similarity / Minor Coincidence"
    else:
        risk = "Clean / Highly Original"

    return {
        "n": n,
        "original_ngrams_count": len(orig_set),
        "rewritten_ngrams_count": len(reph_set),
        "matching_ngrams_count": len(common_ngrams),
        "overlap_percentage": overlap_pct,
        "originality_score": originality,
        "risk_level": risk,
        "matching_sequences": matching_phrases,
        "three_gram_overlap": overlap_3,
        "five_gram_overlap": overlap_5,
    }


def sanitize_ai_markers(text: str) -> str:
    """
    Post-processing sanitizer that wipes out em-dashes, en-dashes used parenthetically,
    double hyphens, zero-width watermarks, and common low-perplexity AI artifacts.
    Guarantees that no em dashes or invisible characters reach the output.
    """
    if not text:
        return ""

    # Replace parenthetical em dashes: "word—phrase—word" -> "word, phrase, word"
    text = re.sub(r"\s*—\s*", ", ", text)
    text = re.sub(r"—", ", ", text)
    text = re.sub(r"\s*--\s*", ", ", text)

    # Strip invisible zero-width Unicode characters / AI watermark tags
    text = re.sub(r"[\u200B-\u200D\uFEFF\u200E\u200F\u202A-\u202E]", "", text)

    # Clean up double commas, comma before period, or comma before semicolon
    text = re.sub(r",\s*,+", ",", text)
    text = re.sub(r",\s*\.", ".", text)
    text = re.sub(r",\s*;", ";", text)

    # Normalize double spaces
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


