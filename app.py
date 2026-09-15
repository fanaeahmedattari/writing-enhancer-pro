"""
app.py — AI Humanizer & Academic Rewriter: Streamlit Frontend
Full-featured web interface with multi-format upload, text paste, AI audit,
multi-provider LLM processing, and multi-format download.
"""

import os
import io
import re
import tempfile
import time
from typing import Any, Dict, List, Optional, Union

import streamlit as st
from dotenv import load_dotenv

from core.docx_parser import DocxChunker
from core.llm_engine import (
    LLMHumanizerEngine,
    LLMProvider,
    AVAILABLE_MODELS,
    DEFAULT_MODELS,
)
from core.prompt_rules import analyze_ai_patterns, calculate_ngram_similarity
from core.unicode_cleaner import sanitize_unicode


# Load .env so keys can be pre-filled
load_dotenv()


def get_secret(key_name: str) -> str:
    """Retrieve secret from Streamlit Cloud secrets or local .env environment variable."""
    try:
        if hasattr(st, "secrets") and key_name in st.secrets:
            return str(st.secrets[key_name])
    except Exception:
        pass
    return os.getenv(key_name, "")


# =========================================================================== #
# Page Config & Custom CSS
# =========================================================================== #

st.set_page_config(
    page_title="Writing Enhancer Pro",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown("""
<style>
    /* Metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea22 0%, #764ba222 100%);
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 12px 16px;
    }
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 20px;
        border-radius: 8px;
    }
    /* Download buttons */
    .stDownloadButton > button {
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

# =========================================================================== #
# Sidebar: Provider, API Key, Model, Options
# =========================================================================== #

st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# --- Provider Selection ---
provider_labels = {
    "gemini": "🔷 Google Gemini (Direct)",
    "openai": "🟢 OpenAI (Direct)",
    "openrouter": "🌐 OpenRouter (Multi-Model)",
}
selected_provider = st.sidebar.selectbox(
    "API Provider",
    options=list(provider_labels.keys()),
    format_func=lambda x: provider_labels[x],
    index=0,
    help="Select your LLM provider. OpenRouter gives access to Claude, GPT, Llama, and more with a single key.",
)

# --- API Key ---
env_key_map = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
env_key = get_secret(env_key_map.get(selected_provider, ""))
if env_key:
    st.sidebar.success("🔒 **Master API Key Active (Server-Secured)**")
    st.sidebar.caption("The app is connected to the pre-configured admin key. Your fellows cannot see, copy, or inspect it.")
    user_override = st.sidebar.text_input(
        "Custom API Key (Optional)",
        value="",
        type="password",
        placeholder="Leave blank to use Server Key...",
        help="Leave empty to use the admin's pre-configured key, or enter your own key to override.",
    )
    api_key = user_override.strip() if user_override.strip() else env_key
else:
    api_key = st.sidebar.text_input(
        "API Key",
        value="",
        type="password",
        placeholder="Paste your Gemini / OpenAI API Key here...",
        help="Your API key is only kept in temporary session memory and never shared.",
    )


# API Keys Guide Expander
with st.sidebar.expander("🔑 API Key Guide (Free & Setup)", expanded=False):
    st.markdown("""
**1. Google Gemini (Recommended - Free):**
- Visit [Google AI Studio](https://aistudio.google.com/apikey).
- Click **"Create API key"**.
- Free tier gives generous limits (15 requests/minute) with **no credit card required**!

**2. OpenRouter (Multi-Model):**
- Visit [openrouter.ai/keys](https://openrouter.ai/keys).
- Access Claude 3.5 Sonnet, GPT-4o, Llama 3.3, and DeepSeek in one account.

**3. OpenAI:**
- Get a key at [platform.openai.com](https://platform.openai.com/api-keys).

**💡 How to save permanently:**
Add to `.env` in the project root:
```env
GEMINI_API_KEY=your_key_here
# or
OPENROUTER_API_KEY=your_key_here
```
Or in **Streamlit Cloud Settings > Secrets**.
""")

# --- Model Selection ---
provider_enum = LLMProvider(selected_provider)
model_list = AVAILABLE_MODELS.get(provider_enum, ["gemini-2.5-flash"])
default_model = DEFAULT_MODELS.get(provider_enum, model_list[0])
default_idx = model_list.index(default_model) if default_model in model_list else 0

selected_model = st.sidebar.selectbox(
    "Model",
    options=model_list,
    index=default_idx,
    key=f"sb_model_select_{selected_provider}",
)

# --- Rewrite Mode ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Rewrite Mode")
aggressive_mode = st.sidebar.toggle(
    "Aggressive Anti-AI Mode",
    value=False,
    help="ON = Maximum watermark disruption + deep token-level rewriting. OFF = Balanced academic enhancement.",
)

# --- User-Selectable Options ---
st.sidebar.markdown("---")
st.sidebar.subheader("📋 Processing Options")
st.sidebar.caption("Select what transformations to apply:")

opt_humanize = st.sidebar.checkbox("✅ Humanize (Remove AI patterns)", value=True)
opt_academic = st.sidebar.checkbox("✅ Academic Tone & Lexicon", value=True)
opt_plagiarism = st.sidebar.checkbox("✅ Anti-Plagiarism (N-Gram Breaking)", value=True)
opt_flow = st.sidebar.checkbox("✅ Flow & Paragraph Stitching", value=True)
opt_structure = st.sidebar.checkbox("✅ Sentence Structure (Burstiness)", value=True)
opt_tables = st.sidebar.checkbox("✅ Tables & Figures Integrity", value=True)

user_options = {
    "humanize": opt_humanize,
    "academic_tone": opt_academic,
    "plagiarism_remover": opt_plagiarism,
    "flow_stitching": opt_flow,
    "sentence_structure": opt_structure,
    "tables_and_figures": opt_tables,
}


# --- Chunk Size ---
st.sidebar.markdown("---")
chunk_size = st.sidebar.slider(
    "Chunk Size (words per block)",
    min_value=200,
    max_value=800,
    value=400,
    step=50,
    help="Lower = more precise, slightly slower. Higher = faster, less granular.",
)

# --- Temperature ---
temperature = st.sidebar.slider(
    "Temperature",
    min_value=0.1,
    max_value=1.0,
    value=0.7,
    step=0.1,
    help="Higher = more creative variation. Lower = closer to original phrasing.",
)

# --- Academic Work Mode & English Tone ---
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Academic Mode & Regional Tone")

opt_work_mode = st.sidebar.selectbox(
    "Work Mode / Document Type",
    options=["journal", "thesis", "assignment", "review_paper", "conference", "grant_proposal", "essay"],
    format_func=lambda x: {
        "journal": "📄 Journal Article (IMRaD Standard)",
        "thesis": "🎓 Thesis / PhD Dissertation",
        "assignment": "📚 Coursework / Student Assignment",
        "review_paper": "📖 Systematic Literature Review",
        "conference": "💻 Conference Paper (IEEE / ACM)",
        "grant_proposal": "💡 Grant Proposal / Scientific Pitch",
        "essay": "📝 Academic Essay / Term Paper",
    }[x],
    index=0,
    help="Customizes structural argumentation and section expectations to match your target document format."
)

opt_english_tone = st.sidebar.selectbox(
    "English Tone & Regional Style",
    options=["academic", "professional", "native_us", "native_uk", "indo_pak"],
    format_func=lambda x: {
        "academic": "🏛️ Academic Rigorous (Global High-Impact)",
        "professional": "💼 Professional & Executive (Crisp & Direct)",
        "native_us": "🇺🇸 Native US English (American Standard - APA)",
        "native_uk": "🇬🇧 Native UK / Oxford English (British Standard)",
        "indo_pak": "🌏 Indo-Pak / Global Scholarly (Idiom Normalizer)",
    }[x],
    index=0,
    help="Indo-Pak mode specifically audits and converts characteristic South Asian regional idioms (e.g. 'revert back', 'do the needful', 'passed out', 'inculcate', 'intimate the committee', 'prepone') into internationally standard native scholarly English."
)

opt_strict_mode = st.sidebar.checkbox(
    "⚡ Strict Academic & Anti-Detection Mode",
    value=False,
    help="Strict ban on em-dashes ('—'), radical sentence burstiness, zero contractions, high epistemic hedging, and Turnitin n-gram shingle decoupling."
)

# --- Academic Formatting & Layout Controls ---
st.sidebar.markdown("---")
st.sidebar.subheader("📐 Document Layout & Formatting")

opt_title_page = st.sidebar.checkbox(
    "🏛️ Include Formal Academic Title Page",
    value=False,
    help="Inserts a publication-grade academic title page before the Table of Contents or body."
)

title_page_dict = None
if opt_title_page:
    with st.sidebar.expander("📝 Title Page Metadata", expanded=True):
        tp_title = st.text_input("Document / Assignment Title", value="Academic Research Manuscript", key="sb_tp_title")
        tp_paper_type = st.selectbox(
            "Document Category",
            options=["Original Research Article", "Student Assignment / Coursework", "Master's Thesis", "PhD Dissertation", "Systematic Review", "Conference Proceedings", "Technical Report", "Term Paper"],
            index=1 if opt_work_mode == "assignment" else 0,
            key="sb_tp_type"
        )
        tp_author = st.text_input("Author / Student Name", value="Author Name", key="sb_tp_author")
        
        # Student Assignment specific fields
        tp_student_id = st.text_input("Student ID / Roll Number (Optional)", value="" if opt_work_mode != "assignment" else "FA22-BCS-089", key="sb_tp_sid")
        tp_course = st.text_input("Course Name / Code (Optional)", value="" if opt_work_mode != "assignment" else "CS-401: Advanced Operating Systems", key="sb_tp_course")
        tp_instructor = st.text_input("Instructor / Professor Name (Optional)", value="" if opt_work_mode != "assignment" else "Prof. Dr. Jane Doe", key="sb_tp_prof")
        
        tp_institution = st.text_input("Institution / Department", value="Department of Computer Science & Engineering", key="sb_tp_inst")
        tp_date = st.text_input("Date / Submission Term", value="September 2026", key="sb_tp_date")
        title_page_dict = {
            "title": tp_title,
            "paper_type": tp_paper_type,
            "author": tp_author,
            "student_id": tp_student_id,
            "course": tp_course,
            "instructor": tp_instructor,
            "institution": tp_institution,
            "date": tp_date,
        }

opt_preserve_media = st.sidebar.checkbox(
    "🖼️ Preserve Original Figures & Media (In-Place)",
    value=True,
    help="Default & Recommended for Theses/Journals: In-place document reassembly preserves 100% of all embedded images, molecular dockings, Western blots, charts, and equations."
)

opt_toc = st.sidebar.checkbox(
    "📑 Generate Table of Contents (TOC)",
    value=True,
    help="Automatically compiles a formal Table of Contents page with leader dots and dynamic Word XML field."
)

opt_margins = st.sidebar.toggle("📄 1-Inch Standard Margins", value=True)
opt_headers = st.sidebar.toggle("🏷️ Running Header & Page Numbers", value=True)

opt_alignment = st.sidebar.selectbox(
    "Paragraph Alignment",
    options=["Justified (Academic Standard)", "Left-aligned", "Centered", "Right-aligned"],
    index=0,
    help="Full justification (both margins aligned) is mandatory in peer-reviewed journals, university theses, and academic assignments."
)
clean_alignment = opt_alignment.split()[0].upper()

opt_typography = st.sidebar.selectbox(
    "Typography Standard",
    options=["Times New Roman (APA / Harvard)", "Calibri (Modern Academic)", "Arial (IEEE Standard)"],
    index=0,
)
opt_font_family = "Times New Roman" if "Times" in opt_typography else ("Calibri" if "Calibri" in opt_typography else "Arial")
opt_line_spacing = st.sidebar.select_slider("Line Spacing", options=[1.15, 1.3, 1.5, 2.0], value=1.5)

# --- Length Strategy for Multi-Page Documents ---
doc_length_mode = st.sidebar.selectbox(
    "Length Strategy",
    options=["preserve", "concise", "elaborate"],
    format_func=lambda x: "Preserve Length (~100%, prevents shrinkage)" if x == "preserve" else ("Tighten & Condense (-25%)" if x == "concise" else "Elaborate & Deepen (+25%)"),
    index=0,
    help="Default 'preserve' ensures that removing AI filler doesn't cause the text to shrink drastically."
)

# =========================================================================== #
# Helper Functions
# =========================================================================== #

def get_engine() -> LLMHumanizerEngine:
    """Create and return an LLMHumanizerEngine from sidebar settings."""
    return LLMHumanizerEngine(
        api_key=api_key,
        provider=selected_provider,
        model_name=selected_model,
        temperature=temperature,
        max_output_tokens=4096,
    )


def render_ai_audit(text: str, label: str = ""):
    """Render AI pattern audit metrics in columns."""
    audit = analyze_ai_patterns(text)
    prefix = f"{label} " if label else ""
    score = audit["score"]

    if score >= 60:
        score_color = "🔴"
    elif score >= 25:
        score_color = "🟡"
    else:
        score_color = "🟢"

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(f"{prefix}AI Score", f"{score_color} {score:.1f}/100")
    c2.metric(f"{prefix}Level", audit["level"])
    c3.metric(f"{prefix}Words", audit["word_count"])
    c4.metric(f"{prefix}Sentences", audit["sentence_count"])

    return audit


def render_detailed_audit(audit: dict):
    """Render a detailed breakdown of AI markers found."""
    with st.expander("🔍 Detailed AI Marker Breakdown", expanded=False):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Vocabulary & Style Issues**")
            items = [
                ("Banned Transitions", audit.get("banned_transitions", [])),
                ("Filler Buzzwords", audit.get("filler_words", [])),
                ("Robotic Qualifiers", audit.get("robotic_qualifiers", [])),
                ("Leak Tokens", audit.get("leak_tokens", [])),
            ]
            for label, found in items:
                count = len(found)
                icon = "🔴" if count > 0 else "✅"
                st.markdown(f"{icon} **{label}:** {count}")
                if found:
                    st.code(", ".join(found[:10]), language=None)

        with col_b:
            st.markdown("**Structural & Tone Issues**")
            struct_items = [
                ("Negative Parallelisms", audit.get("negative_parallelisms", 0)),
                ("Trailing Participles", audit.get("trailing_participles", 0)),
                ("Em Dashes", audit.get("em_dashes_count", 0)),
                ("Contractions", audit.get("contractions_count", 0)),
                ("1st/2nd Person", audit.get("first_second_person_count", 0)),
                ("Exclamation Marks", audit.get("exclamations_count", 0)),
                ("Subjective Opinions", audit.get("subjective_opinions_count", 0)),
            ]
            for label, count in struct_items:
                icon = "🔴" if count > 0 else "✅"
                st.markdown(f"{icon} **{label}:** {count}")

        st.markdown(f"**Avg Sentence Length:** {audit.get('avg_sentence_length', 0)} words")


def text_to_docx_bytes(
    text: str,
    typography_preset: str = "Times New Roman",
    margins_inches: float = 1.0,
    line_spacing: float = 1.5,
    include_title_page: bool = False,
    title_page_data: Optional[Dict[str, str]] = None,
    include_toc: bool = True,
    running_head: str = "Writing Enhancer Pro — Academic Manuscript",
    alignment: str = "JUSTIFY",
) -> bytes:
    """Convert text to a beautifully styled .docx with 1-inch margins, title page, academic typography, full justification, and page numbers."""
    import tempfile
    from core.docx_parser import DocxChunker
    chunker = DocxChunker()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
        tmp_path = tmp.name
    try:
        chunker.reassemble_docx(
            original_docx_path=tmp_path,
            humanized_chunks=[text],
            output_path=tmp_path,
            include_title_page=include_title_page,
            title_page_data=title_page_data,
            include_toc=include_toc,
            typography_preset=typography_preset,
            margins_inches=margins_inches,
            line_spacing=line_spacing,
            running_head=running_head,
            alignment=alignment,
        )
        with open(tmp_path, "rb") as f:
            data = f.read()
        return data
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def text_to_pdf_bytes(
    text: str,
    title: str = "Enhanced Academic Document",
    include_title_page: bool = False,
    title_page_data: Optional[Dict[str, str]] = None,
    include_toc: bool = True,
    running_head: str = "Writing Enhancer Pro — Academic Manuscript",
    alignment: str = "JUSTIFY",
) -> bytes:
    """Compile text into a publication-ready PDF with full justification, 1-inch margins, title page, TOC, and running headers/footers."""
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT, TA_CENTER, TA_RIGHT
    from core.docx_parser import FIGURE_CAPTION_RE, TABLE_CAPTION_RE

    align_dict = {
        "JUSTIFY": TA_JUSTIFY,
        "JUSTIFIED": TA_JUSTIFY,
        "LEFT": TA_LEFT,
        "CENTER": TA_CENTER,
        "RIGHT": TA_RIGHT,
    }
    clean_align = alignment.strip().upper().split()[0]
    target_pdf_align = align_dict.get(clean_align, TA_JUSTIFY)

    buf = io.BytesIO()
    # 72 points = standard 1.0 inch academic margin
    doc = SimpleDocTemplate(
        buf,
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72,
    )
    styles = getSampleStyleSheet()

    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10.5,
        leading=16,
        textColor=colors.HexColor("#212121"),
        spaceAfter=10,
        alignment=target_pdf_align,
    )
    fig_caption_style = ParagraphStyle(
        "FigCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=9.5,
        leading=14,
        textColor=colors.HexColor("#555555"),
        alignment=1,  # Centered
        spaceBefore=4,
        spaceAfter=12,
    )
    tbl_caption_style = ParagraphStyle(
        "TblCaption",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10.5,
        leading=15,
        textColor=colors.HexColor("#1a237e"),
        spaceBefore=12,
        spaceAfter=4,
        keepWithNext=True,
    )
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=14,
        alignment=0,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=13.5,
        leading=18,
        textColor=colors.HexColor("#1a237e"),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True,
    )
    toc_title_style = ParagraphStyle(
        "TOCTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=16,
        leading=20,
        textColor=colors.HexColor("#1a237e"),
        spaceAfter=12,
        alignment=1,  # Centered
    )
    toc_entry_style = ParagraphStyle(
        "TOCEntry",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#333333"),
        spaceAfter=4,
    )

    story = []

    # Title Page Generation if requested
    if include_title_page:
        data = title_page_data or {}
        tp_title_txt = data.get("title", "").strip() or title
        tp_type_txt = data.get("paper_type", "").strip() or "Research Paper"
        tp_auth_txt = data.get("author", "").strip() or "Author Name"
        tp_student_id = data.get("student_id", "").strip()
        tp_course = data.get("course", "").strip()
        tp_instructor = data.get("instructor", "").strip()
        tp_inst_txt = data.get("institution", "").strip() or "Academic Institution"
        tp_date_txt = data.get("date", "").strip() or "September 2026"

        tp_title_style = ParagraphStyle("TPTitle", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22, leading=26, textColor=colors.HexColor("#1a237e"), alignment=1, spaceBefore=70, spaceAfter=14)
        tp_sub_style = ParagraphStyle("TPSub", parent=styles["Normal"], fontName="Helvetica-Oblique", fontSize=13, leading=17, textColor=colors.HexColor("#555555"), alignment=1, spaceAfter=20)
        tp_course_style = ParagraphStyle("TPCourse", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=12, leading=16, textColor=colors.HexColor("#283593"), alignment=1, spaceAfter=14)
        tp_auth_style = ParagraphStyle("TPAuth", parent=styles["Normal"], fontName="Helvetica-Bold", fontSize=13.5, leading=18, textColor=colors.HexColor("#212121"), alignment=1, spaceAfter=6)
        tp_inst_detail_style = ParagraphStyle("TPInstDetail", parent=styles["Normal"], fontName="Helvetica", fontSize=11.5, leading=15, textColor=colors.HexColor("#444444"), alignment=1, spaceAfter=6)
        tp_inst_style = ParagraphStyle("TPInst", parent=styles["Normal"], fontName="Helvetica", fontSize=11.5, leading=15, textColor=colors.HexColor("#444444"), alignment=1, spaceAfter=6)
        tp_date_style = ParagraphStyle("TPDate", parent=styles["Normal"], fontName="Helvetica", fontSize=11, leading=15, textColor=colors.HexColor("#777777"), alignment=1, spaceAfter=70)

        story.append(Paragraph(tp_title_txt, tp_title_style))
        story.append(Paragraph(tp_type_txt, tp_sub_style))
        if tp_course:
            story.append(Paragraph(f"<b>Course:</b> {tp_course}", tp_course_style))
        auth_line = f"<b>{tp_auth_txt}</b> (Student ID / Roll No: {tp_student_id})" if tp_student_id else f"<b>{tp_auth_txt}</b>"
        story.append(Paragraph(auth_line, tp_auth_style))
        if tp_instructor:
            story.append(Paragraph(f"Submitted to: {tp_instructor}", tp_inst_detail_style))
        story.append(Paragraph(tp_inst_txt, tp_inst_style))
        story.append(Paragraph(tp_date_txt, tp_date_style))
        story.append(PageBreak())
    else:
        story.append(Paragraph(title, title_style))
        story.append(Spacer(1, 10))

    # Detect headings for Table of Contents
    blocks = [b.strip() for b in text.split("\n\n") if b.strip()]
    headings = []
    for b in blocks:
        hm = re.match(r"^(#{1,4})\s+(.*)$", b)
        if hm:
            headings.append((len(hm.group(1)), hm.group(2).strip()))

    # Insert Table of Contents if requested and multiple headings exist
    if include_toc and len(headings) >= 2:
        story.append(Spacer(1, 8))
        story.append(Paragraph("Table of Contents", toc_title_style))
        story.append(Spacer(1, 6))

        est_p = 3 if include_title_page else 2
        for lvl, h_txt in headings:
            indent_str = "&nbsp;&nbsp;&nbsp;&nbsp;" * (lvl - 1)
            dots = "." * max(6, 54 - len(h_txt) - (lvl * 4))
            entry_html = f"<b>{indent_str}{h_txt}</b> <font color='#9e9e9e'>{dots}</font> <font color='#616161'>Page {est_p}</font>"
            story.append(Paragraph(entry_html, toc_entry_style))
            if lvl == 1:
                est_p += 1

        story.append(PageBreak())

    # Main content blocks
    for block in blocks:
        if block.startswith("# "):
            story.append(Paragraph(block[2:].strip(), heading_style))
        elif block.startswith("## "):
            story.append(Paragraph(block[3:].strip(), heading_style))
        elif block.startswith("### "):
            story.append(Paragraph(block[4:].strip(), heading_style))
        elif block.startswith("#### "):
            story.append(Paragraph(block[5:].strip(), heading_style))
        elif FIGURE_CAPTION_RE.match(block):
            story.append(Paragraph(block.strip(), fig_caption_style))
        elif TABLE_CAPTION_RE.match(block):
            story.append(Paragraph(block.strip(), tbl_caption_style))
        else:
            safe = block.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace("\n", "<br/>")
            story.append(Paragraph(safe, body_style))

    # Running header and page numbering footer
    def draw_decorations(canvas, d):
        canvas.saveState()
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#757575"))
        canvas.drawRightString(d.pagesize[0] - 72, d.pagesize[1] - 45, running_head)
        canvas.drawCentredString(d.pagesize[0] / 2.0, 42, f"Page {canvas.getPageNumber()}")
        canvas.restoreState()

    doc.build(story, onFirstPage=draw_decorations, onLaterPages=draw_decorations)
    return buf.getvalue()


def extract_text_from_pdf(pdf_source) -> str:
    """Extract plain text from uploaded PDF bytes or file-like object using pypdf."""
    from pypdf import PdfReader
    if isinstance(pdf_source, (bytes, bytearray)):
        stream = io.BytesIO(pdf_source)
    elif hasattr(pdf_source, "read"):
        if hasattr(pdf_source, "seek"):
            pdf_source.seek(0)
        stream = pdf_source
    else:
        stream = pdf_source
    reader = PdfReader(stream)
    extracted = []
    for page in reader.pages:
        t = page.extract_text()
        if t and t.strip():
            extracted.append(t.strip())
    return "\n\n".join(extracted)


def extract_text_from_docx_bytes(docx_bytes: bytes) -> str:
    """Extract full structured text from .docx bytes including paragraphs and tables."""
    import docx
    doc = docx.Document(io.BytesIO(docx_bytes))
    full_text = []
    for para in doc.paragraphs:
        if para.text.strip():
            full_text.append(para.text.strip())
    for tbl in doc.tables:
        for row in tbl.rows:
            row_text = " | ".join(c.text.strip() for c in row.cells if c.text.strip())
            if row_text:
                full_text.append(f"| {row_text} |")
    return "\n\n".join(full_text)


def extract_text_from_pptx(pptx_source) -> str:
    """Extract structured plain text, slide titles, tables, and speaker notes from PowerPoint (.pptx)."""
    from pptx import Presentation
    if isinstance(pptx_source, (bytes, bytearray)):
        stream = io.BytesIO(pptx_source)
    elif hasattr(pptx_source, "read"):
        if hasattr(pptx_source, "seek"):
            pptx_source.seek(0)
        stream = io.BytesIO(pptx_source.read())
    else:
        stream = pptx_source

    prs = Presentation(stream)
    slide_chunks = []
    for idx, slide in enumerate(prs.slides, start=1):
        slide_lines = []
        if slide.shapes.title and slide.shapes.title.text.strip():
            title_text = slide.shapes.title.text.strip().replace("\n", " ")
            slide_lines.append(f"# Slide {idx}: {title_text}")
        else:
            slide_lines.append(f"# Slide {idx}")

        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if shape.has_text_frame:
                for p in shape.text_frame.paragraphs:
                    ptxt = "".join(r.text for r in p.runs).strip()
                    if ptxt:
                        slide_lines.append(ptxt)
            elif shape.has_table:
                tbl_lines = []
                for row in shape.table.rows:
                    cells = [c.text.strip().replace("\n", " ") for c in row.cells]
                    tbl_lines.append("| " + " | ".join(cells) + " |")
                if tbl_lines:
                    sep = "| " + " | ".join(["---"] * len(shape.table.columns)) + " |"
                    slide_lines.append(tbl_lines[0] + "\n" + sep + "\n" + "\n".join(tbl_lines[1:]))

        if slide.has_notes_slide and slide.notes_slide.notes_text_frame:
            notes = slide.notes_slide.notes_text_frame.text.strip()
            if notes:
                slide_lines.append(f"> **Speaker Notes (Slide {idx}):** {notes}")

        if slide_lines:
            slide_chunks.append("\n\n".join(slide_lines))

    return "\n\n---\n\n".join(slide_chunks)


def text_to_pptx_bytes(markdown_text: str, presentation_title: str = "Enhanced Presentation") -> bytes:
    """Generate a clean PowerPoint (.pptx) file from markdown slides separated by --- or # Slide."""
    import re as _re
    from pptx import Presentation
    prs = Presentation()
    title_slide_layout = prs.slide_layouts[0]
    bullet_slide_layout = prs.slide_layouts[1]

    raw_slides = _re.split(r"(?m)^---\s*$", markdown_text)
    if len(raw_slides) <= 1:
        raw_slides = _re.split(r"(?m)^(?=#\s+Slide\s+\d+|#\s+)", markdown_text)
        raw_slides = [s for s in raw_slides if s.strip()]

    if not raw_slides:
        raw_slides = [markdown_text]

    for idx, slide_block in enumerate(raw_slides):
        lines = [line.strip() for line in slide_block.strip().split("\n") if line.strip()]
        if not lines:
            continue

        slide_title = f"Slide {idx + 1}"
        body_lines = []
        for line in lines:
            if line.startswith("#"):
                slide_title = line.lstrip("#").strip()
            else:
                body_lines.append(line)

        if idx == 0 and len(raw_slides) > 1:
            slide = prs.slides.add_slide(title_slide_layout)
            slide.shapes.title.text = slide_title
            if body_lines and len(slide.placeholders) > 1:
                slide.placeholders[1].text = "\n".join(body_lines[:3])
        else:
            slide = prs.slides.add_slide(bullet_slide_layout)
            slide.shapes.title.text = slide_title
            if len(slide.placeholders) > 1:
                tf = slide.placeholders[1].text_frame
                tf.word_wrap = True
                for b_idx, bl in enumerate(body_lines):
                    clean_b = bl.lstrip("*-• ")
                    if b_idx == 0:
                        tf.text = clean_b
                    else:
                        p = tf.add_paragraph()
                        p.text = clean_b
                        p.level = 0

    buf = io.BytesIO()
    prs.save(buf)
    return buf.getvalue()


def generate_source_citation_entry(filename: str, source_text: str, index: int, style: str = "APA 7th Edition") -> Dict[str, str]:
    """Generates an academic in-text citation and bibliographic reference entry from a source file."""
    import re as _re
    # Clean base name
    raw_name = filename.rsplit(".", 1)[0].replace("_", " ").replace("-", " ").strip()
    
    # Try to find a 4-digit year in filename or first 300 chars of text
    year_match = _re.search(r"(?:^|[^0-9])(19\d\d|20\d\d)(?:[^0-9]|$)", filename)
    if not year_match:
        year_match = _re.search(r"(?:^|[^0-9])(19\d\d|20\d\d)(?:[^0-9]|$)", source_text[:350])
    year = year_match.group(1) if year_match else "2024"

    # Extract first line as potential title
    first_lines = [l.strip().lstrip("#").strip() for l in source_text[:400].split("\n") if l.strip()]
    doc_title = first_lines[0] if first_lines and len(first_lines[0]) < 120 else raw_name.title()

    # Formulate author from first token of filename
    clean_author = raw_name.split()[0].title() if raw_name.split() else "Author"
    
    if "IEEE" in style:
        in_text = f"[{index}]"
        bib_entry = f"[{index}] {clean_author} et al., \"{doc_title},\" {year}."
    elif "Harvard" in style:
        in_text = f"({clean_author}, {year})"
        bib_entry = f"{clean_author} ({year}) '{doc_title}'. Available at: {filename}."
    else:  # APA 7th Edition
        in_text = f"({clean_author} et al., {year})"
        bib_entry = f"{clean_author}, A. et al. ({year}). {doc_title}. Reference source: {filename}."

    return {
        "in_text": in_text,
        "bib_entry": bib_entry,
        "title": doc_title,
        "year": year,
        "author": clean_author,
        "filename": filename
    }


def generate_inline_diff_html(original: str, modified: str) -> str:
    """
    Generates rich HTML showing word-by-word visual differences:
    - Red strikethrough for deleted AI words/phrases
    - Green highlight for added scholarly/humanized words
    """
    import difflib

    orig_words = original.split()
    mod_words = modified.split()

    matcher = difflib.SequenceMatcher(None, orig_words, mod_words)
    has_differences = any(tag in ("delete", "insert", "replace") for tag, _, _, _, _ in matcher.get_opcodes())
    html_chunks = []

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            html_chunks.append(f'<span style="color:#212529;">{" ".join(orig_words[i1:i2])}</span>')
        elif tag == "delete":
            deleted_text = " ".join(orig_words[i1:i2])
            html_chunks.append(
                f'<span style="color:#b31d28; background-color:#ffebe9; text-decoration:line-through; text-decoration-color:#b31d28; padding:2px 5px; margin:0 1px; border-radius:4px; font-weight:600;">{deleted_text}</span>'
            )
        elif tag == "insert":
            inserted_text = " ".join(mod_words[j1:j2])
            html_chunks.append(
                f'<span style="color:#116329; background-color:#dafbe1; padding:2px 5px; margin:0 1px; border-radius:4px; font-weight:600;">{inserted_text}</span>'
            )
        elif tag == "replace":
            deleted_text = " ".join(orig_words[i1:i2])
            inserted_text = " ".join(mod_words[j1:j2])
            html_chunks.append(
                f'<span style="color:#b31d28; background-color:#ffebe9; text-decoration:line-through; text-decoration-color:#b31d28; padding:2px 5px; margin:0 1px; border-radius:4px; font-weight:600;">{deleted_text}</span> '
                f'<span style="color:#116329; background-color:#dafbe1; padding:2px 5px; margin:0 1px; border-radius:4px; font-weight:600;">{inserted_text}</span>'
            )

    diff_body = " ".join(html_chunks).replace("\n", "<br/>")

    legend = (
        '<div style="display:flex; flex-wrap:wrap; gap:16px; margin-bottom:12px; font-size:0.90em; font-weight:600; padding:8px 12px; background:#f6f8fa; border-radius:6px; border:1px solid #d0d7de; color:#24292f;">'
        '<span><span style="display:inline-block; width:14px; height:14px; background:#ffebe9; border:1px solid #b31d28; border-radius:3px; margin-right:5px; vertical-align:middle;"></span> <span style="color:#b31d28; text-decoration:line-through;">Red Strikethrough</span>: Removed AI / Source Phrasing</span>'
        '<span><span style="display:inline-block; width:14px; height:14px; background:#dafbe1; border:1px solid #116329; border-radius:3px; margin-right:5px; vertical-align:middle;"></span> <span style="color:#116329;">Green Highlight</span>: Enhanced Scholarly Replacement</span>'
        '</div>'
    )

    notice = ""
    if not has_differences:
        notice = '<div style="background-color: #fff3cd; color: #856404; padding: 12px 16px; border-radius: 6px; margin-bottom: 12px; font-size: 0.95em; font-weight: 500; border-left: 5px solid #ffeeba;">⚠️ <strong>Notice:</strong> Zero word differences detected between original and output text. Verify that your API key is valid and an enhancement mode is active.</div>'

    return f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.8; font-size: 1.0em; padding: 16px; border: 1px solid #d0d7de; border-radius: 8px; background-color: #ffffff; color: #24292f; max-height: 480px; overflow-y: auto;">
        {legend}
        {notice}
        {diff_body}
    </div>
    """



# =========================================================================== #
# Main UI: Title & Tabs
# =========================================================================== #

st.title("✍️ Writing Enhancer Pro")
st.caption("All-in-One Scientific & Academic Writing Suite: AI Humanizer, Tone & Flow Enhancer, Deep Paraphraser & Plagiarism Reducer")

if not api_key:
    st.warning("🔑 **API Key Required:** Please enter your Google Gemini API Key in the left sidebar to enable rewriting and humanization (or configure `GEMINI_API_KEY` in Streamlit Cloud Secrets).")

tab_doc, tab_text, tab_plagiarism, tab_audit, tab_help = st.tabs([
    "📄 Document Enhancer",
    "✏️ Quick Text Rewriter",
    "🛡️ Plagiarism & Paraphrasing Studio",
    "🔬 AI Marker Auditor",
    "📖 User Guide & Help",
])



# =========================================================================== #
# TAB 1: Document Enhancer (File Upload)
# =========================================================================== #

with tab_doc:
    st.markdown("### Upload your document for full processing")

    st.caption("Supported formats: Word Document (.docx), Compiled PDF (.pdf), PowerPoint Presentation (.pptx), Plain Text (.txt), Markdown (.md)")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["docx", "pdf", "pptx", "txt", "md"],
        help="Supported formats: Word Document (.docx), PDF (.pdf), PowerPoint (.pptx), Plain Text (.txt), Markdown (.md)",
        key="doc_uploader",
    )

    if uploaded_file is not None:
        st.success(f"📁 **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")

        # Save or convert to temp .docx
        if uploaded_file.name.lower().endswith(".pdf"):
            with st.spinner("Extracting text from PDF..."):
                raw_content = extract_text_from_pdf(uploaded_file.getvalue())
            import docx as _docx
            doc = _docx.Document()
            for block in raw_content.split("\n\n"):
                clean_b = block.strip()
                if clean_b:
                    doc.add_paragraph(clean_b)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc.save(tmp.name)
                input_path = tmp.name
        elif uploaded_file.name.lower().endswith(".pptx"):
            with st.spinner("Extracting slides and notes from PowerPoint (.pptx)..."):
                raw_content = extract_text_from_pptx(uploaded_file.getvalue())
            import docx as _docx
            doc = _docx.Document()
            for block in raw_content.split("\n\n"):
                clean_b = block.strip()
                if clean_b.startswith("# "):
                    doc.add_heading(clean_b[2:].strip(), level=1)
                elif clean_b.startswith("## "):
                    doc.add_heading(clean_b[3:].strip(), level=2)
                elif clean_b.startswith("### "):
                    doc.add_heading(clean_b[4:].strip(), level=3)
                elif clean_b:
                    doc.add_paragraph(clean_b)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc.save(tmp.name)
                input_path = tmp.name
        elif uploaded_file.name.lower().endswith((".txt", ".md")):
            raw_content = uploaded_file.getvalue().decode("utf-8", errors="ignore")
            import docx as _docx
            doc = _docx.Document()
            for block in raw_content.split("\n\n"):
                clean_b = block.strip()
                if clean_b.startswith("# "):
                    doc.add_heading(clean_b[2:].strip(), level=1)
                elif clean_b.startswith("## "):
                    doc.add_heading(clean_b[3:].strip(), level=2)
                elif clean_b.startswith("### "):
                    doc.add_heading(clean_b[4:].strip(), level=3)
                elif clean_b:
                    doc.add_paragraph(clean_b)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                doc.save(tmp.name)
                input_path = tmp.name
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                tmp.write(uploaded_file.getvalue())
                input_path = tmp.name

        # Parse & pre-scan
        chunker = DocxChunker(max_words_per_chunk=chunk_size)

        with st.spinner("Parsing document structure..."):
            chunks = chunker.parse_docx(input_path)
            try:
                fig_tbl_scan = chunker.scan_figures_and_tables(input_path)
            except Exception as _scan_err:
                fig_tbl_scan = {
                    "total_tables": 0,
                    "tables_structure": [],
                    "total_figure_captions": 0,
                    "figure_captions": [],
                    "total_table_captions": 0,
                    "table_captions": [],
                    "in_text_figure_citations": [],
                    "in_text_table_citations": [],
                    "missing_figure_citations": [],
                    "missing_table_citations": [],
                    "uncaptioned_figure_references": [],
                    "uncaptioned_table_references": [],
                    "academic_advice": []
                }

        # Document overview
        total_words = sum(c["word_count"] for c in chunks)
        sections_found = list({c["section_type"] for c in chunks})

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Chunks", len(chunks))
        col2.metric("Total Words", f"{total_words:,}")
        col3.metric("Tables Found", fig_tbl_scan["total_tables"])
        col4.metric("Figure Captions", fig_tbl_scan["total_figure_captions"])

        st.markdown(f"**Detected Sections:** {', '.join(s.replace('_', ' ').title() for s in sections_found)}")

        # Academic advice from figure/table scan
        if fig_tbl_scan["academic_advice"]:
            for advice in fig_tbl_scan["academic_advice"]:
                st.warning(f"📊 {advice}")

        # Pre-scan AI markers on full document text
        full_text = "\n\n".join(c["text"] for c in chunks)
        st.markdown("#### Pre-Processing AI Marker Scan")
        pre_audit = render_ai_audit(full_text, label="Before")
        render_detailed_audit(pre_audit)

        st.markdown("---")

        # Document Processing Mode
        doc_proc_mode = st.radio(
            "Select Processing Mode:",
            options=[
                "✨ Full Academic Enhancement & AI De-Watermarking",
                "🛡️ Deep Paraphraser & N-Gram Disrupter",
                "📐 Format Document Only (Instant Academic Typesetting — Zero API / Zero Token Cost)"
            ],
            index=0,
            horizontal=True,
            help="'Format Document Only' typesets your file with Title Page, Table of Contents, 1-inch margins, typography, running headers, and page numbers instantly without modifying text or calling any LLM."
        )

        if "Format Document Only" in doc_proc_mode:
            st.info("💡 **Instant Academic Typesetting:** Formats your uploaded document's layout with the sidebar-selected Title Page, Table of Contents, 1-Inch Margins, Typography, Full Paragraph Justification, Running Header, and Dynamic Page Numbers. Zero API key required, zero token cost!")

            # Step 1: Pre-Analysis display
            with st.expander("🔍 Step 1: Structural & Caption Pre-Analysis", expanded=False):
                from core.docx_parser import analyze_document_structure
                struct_info = analyze_document_structure(full_text)
                ca, cb, cc, cd = st.columns(4)
                ca.metric("Headings Found", struct_info["heading_count"])
                cb.metric("Figure Captions", struct_info["figure_count"])
                cc.metric("Table Captions", struct_info["table_caption_count"])
                cd.metric("Est. Formatted Pages", struct_info["estimated_pages"])

            if st.button("⚡ Format & Typeset Document (Instant)", type="primary", use_container_width=True, key="doc_format_only_btn"):
                with st.status("Executing Step-by-Step Academic Formatting...", expanded=True) as status_box:
                    st.write("Step 1: 🔍 Analyzing document structure, figures, and table captions...")
                    from core.docx_parser import analyze_document_structure
                    struct_info = analyze_document_structure(full_text)
                    st.write(f"Detected {struct_info['heading_count']} headings, {struct_info['figure_count']} figure captions, {struct_info['table_caption_count']} table captions.")

                    st.write(f"Step 2: 📐 Applying '{opt_alignment}' paragraph alignment, 1.0-inch margins, and typography...")
                    raw_texts = [c["text"] for c in chunks]
                    output_fmt_docx = input_path.replace(".docx", "_formatted.docx")
                    chunker.reassemble_docx(
                        original_docx_path=input_path,
                        humanized_chunks=raw_texts,
                        output_path=output_fmt_docx,
                        include_title_page=opt_title_page,
                        title_page_data=title_page_dict,
                        include_toc=opt_toc,
                        typography_preset=opt_font_family,
                        margins_inches=1.0 if opt_margins else 0.75,
                        line_spacing=opt_line_spacing,
                        alignment=clean_alignment,
                    )
                    with open(output_fmt_docx, "rb") as f:
                        fmt_docx_data = f.read()

                    st.write("Step 3: 📑 Compiling Table of Contents and Title Page...")
                    fmt_pdf_data = text_to_pdf_bytes(
                        text=full_text,
                        title=title_page_dict.get("title", "Academic Manuscript") if opt_title_page else uploaded_file.name.rsplit(".", 1)[0].replace("_", " ").title(),
                        include_title_page=opt_title_page,
                        title_page_data=title_page_dict,
                        include_toc=opt_toc,
                        alignment=clean_alignment,
                    )

                    st.write("Step 4: 🛡️ Sanitizing container metadata and finalizing...")
                    status_box.update(label="✅ Document successfully typeset and formatted!", state="complete", expanded=False)

                    st.session_state["tab1_format_results"] = {
                        "docx_data": fmt_docx_data,
                        "pdf_data": fmt_pdf_data,
                        "base_name": uploaded_file.name.rsplit('.', 1)[0],
                    }

            if "tab1_format_results" in st.session_state:
                res_fmt = st.session_state["tab1_format_results"]
                st.success("🎉 Document successfully typeset and formatted! Download in your desired formats below:")
                c_d1, c_d2 = st.columns(2)
                with c_d1:
                    st.download_button("📄 Download Formatted .docx", data=res_fmt["docx_data"], file_name=f"formatted_{res_fmt['base_name']}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
                with c_d2:
                    st.download_button("📕 Download Formatted .pdf", data=res_fmt["pdf_data"], file_name=f"formatted_{res_fmt['base_name']}.pdf", mime="application/pdf", use_container_width=True)
                if st.button("🔄 Reset / Format Again", key="reset_tab1_fmt"):
                    del st.session_state["tab1_format_results"]
                    st.rerun()
        else:
            # Active options display
            active_opts = [k.replace("_", " ").title() for k, v in user_options.items() if v]
            mode_label = "🔴 Aggressive" if (aggressive_mode or "Paraphraser" in doc_proc_mode) else "🟢 Balanced"
            st.markdown(f"**Mode:** {mode_label} &nbsp;|&nbsp; **Active:** {', '.join(active_opts) if active_opts else 'None selected'}")

            # Process button
            if st.button("🚀 Enhance Document", type="primary", use_container_width=True, key="doc_process"):
                if not api_key:
                    st.error("⚠️ Please enter your API Key in the sidebar.")
                elif not any(user_options.values()):
                    st.error("⚠️ Please select at least one processing option in the sidebar.")
                else:
                    try:
                        engine = get_engine()

                        progress_bar = st.progress(0, text="Initializing...")
                        status_area = st.empty()

                        def doc_progress(current, total, message):
                            pct = current / total
                            progress_bar.progress(pct, text=message)

                        results = engine.process_all_chunks(
                            chunks=chunks,
                            aggressive=True if "Paraphraser" in doc_proc_mode else aggressive_mode,
                            options=user_options,
                            length_mode=doc_length_mode,
                            work_mode=opt_work_mode,
                            english_tone=opt_english_tone,
                            strict_mode=opt_strict_mode,
                            progress_callback=doc_progress,
                            inter_chunk_delay=1.0,
                        )

                        progress_bar.progress(1.0, text="✅ Processing complete!")

                        # Count successes/errors
                        successes = sum(1 for r in results if r["status"] == "success")
                        errors = sum(1 for r in results if r["status"] != "success")

                        if errors > 0:
                            st.error(f"❌ {errors} chunk(s) failed to enhance due to API/provider errors and remained as original text.")

                        # Assemble humanized texts
                        humanized_texts = [r["humanized_text"] for r in results]
                        full_humanized = "\n\n".join(humanized_texts)

                        # Post-scan
                        post_audit = render_ai_audit(full_humanized, label="After")

                        # Score delta
                        delta = pre_audit["score"] - post_audit["score"]

                        # Plagiarism & N-Gram Overlap Scan
                        ngram_res = calculate_ngram_similarity(full_text, full_humanized, n=4)

                        # Scientific Data & Numerical Integrity Audit
                        from core.data_integrity import audit_scientific_fidelity
                        fidelity_res = audit_scientific_fidelity(full_text, full_humanized)

                        # Prepare and cache all download formats
                        output_path = input_path.replace(".docx", "_enhanced.docx")
                        chunker.reassemble_docx(
                            original_docx_path=input_path,
                            humanized_chunks=results,
                            output_path=output_path,
                            preserve_original_media=opt_preserve_media,
                            include_title_page=opt_title_page,
                            title_page_data=title_page_dict,
                            include_toc=opt_toc,
                            typography_preset=opt_font_family,
                            margins_inches=1.0 if opt_margins else 0.75,
                            line_spacing=opt_line_spacing,
                            alignment=clean_alignment,
                        )
                        with open(output_path, "rb") as f:
                            docx_bytes = f.read()
                        try:
                            os.remove(output_path)
                        except OSError:
                            pass

                        pdf_bytes = text_to_pdf_bytes(
                            full_humanized,
                            title=title_page_dict.get("title", f"Enhanced: {uploaded_file.name.rsplit('.', 1)[0]}") if opt_title_page else f"Enhanced: {uploaded_file.name.rsplit('.', 1)[0]}",
                            include_title_page=opt_title_page,
                            title_page_data=title_page_dict,
                            include_toc=opt_toc,
                            alignment=clean_alignment,
                        )

                        import re as _re
                        plain_text = _re.sub(r"^#{1,6}\s+", "", full_humanized, flags=_re.MULTILINE)

                        st.session_state["tab1_enhance_results"] = {
                            "docx_bytes": docx_bytes,
                            "pdf_bytes": pdf_bytes,
                            "md_bytes": full_humanized.encode("utf-8"),
                            "txt_bytes": plain_text.encode("utf-8"),
                            "base_name": uploaded_file.name.rsplit('.', 1)[0],
                            "delta": delta,
                            "pre_score": pre_audit['score'],
                            "post_score": post_audit['score'],
                            "post_audit": post_audit,
                            "ngram_res": ngram_res,
                            "fidelity_res": fidelity_res,
                            "results": results,
                            "errors": errors,
                        }
                        st.rerun()

                    except Exception as e:
                        st.error(f"❌ Processing failed: {str(e)}")
                    finally:
                        # Cleanup temp input
                        try:
                            os.remove(input_path)
                        except OSError:
                            pass

        # Persistent Display of Enhanced Results & Downloads (Never disappears upon clicking download or interaction)
        if "tab1_enhance_results" in st.session_state:
            t1_res = st.session_state["tab1_enhance_results"]
            res_list = t1_res.get("results", [])

            st.markdown("---")
            st.markdown("### 📊 Enhancement Results")

            if t1_res.get("errors", 0) > 0:
                st.error(f"❌ {t1_res['errors']} chunk(s) failed to enhance due to API/provider errors and remained as original text.")
                with st.expander("🔍 View Error Details", expanded=False):
                    for r in res_list:
                        if r["status"] != "success":
                            st.markdown(f"- **Chunk {r['chunk_id']} ({r['section_type']})**: `{r['status']}`")

            # Score delta banner
            if t1_res.get("delta", 0) > 0:
                st.success(f"📉 AI Score reduced by **{t1_res['delta']:.1f}** points ({t1_res['pre_score']:.1f} → {t1_res['post_score']:.1f})")

            # Post-scan audit
            st.markdown("#### Post-Processing AI Marker Scan")
            if "post_audit" in t1_res:
                render_detailed_audit(t1_res["post_audit"])

            # Plagiarism & N-Gram Overlap Scan
            ngram_res = t1_res.get("ngram_res", {})
            if ngram_res:
                st.markdown("#### 🛡️ Plagiarism & N-Gram Overlap Scan")
                ng_c1, ng_c2, ng_c3 = st.columns(3)
                ng_c1.metric("Originality Score", f"🛡️ {ngram_res.get('originality_score', 100)}%")
                ng_c2.metric("4-Gram Overlap", f"{ngram_res.get('overlap_percentage', 0)}%")
                ng_c3.metric("Plagiarism Risk", ngram_res.get("risk_level", "Low"))
                if ngram_res.get("matching_sequences"):
                    with st.expander(f"⚠️ Matching 4-Gram Sequences ({len(ngram_res['matching_sequences'])})", expanded=False):
                        st.caption("These exact 4-word sequences match the source text and could be flagged by Turnitin / QuillBot:")
                        st.write(", ".join([f"`{seq}`" for seq in ngram_res["matching_sequences"][:25]]))

            # Scientific Data & Figure Integrity Audit
            fidelity_res = t1_res.get("fidelity_res", {})
            if fidelity_res:
                st.markdown("#### 🔬 Scientific Data & Figure Integrity Audit")
                fid_c1, fid_c2, fid_c3 = st.columns(3)
                fid_score = fidelity_res.get("fidelity_score", 100.0)
                fid_c1.metric("Fidelity Score", f"🔬 {fid_score}%")
                passed = fidelity_res.get("passed_checks", 0)
                total = fidelity_res.get("total_checks", 0)
                fid_c2.metric("Verified Scientific Metrics", f"{passed}/{total}")
                is_safe = fidelity_res.get("is_safe_for_academic_submission", True)
                fid_c3.metric("Academic Submission Safety", "✅ Verified Safe" if is_safe else "⚠️ Review Discrepancies")

                discrepancies = fidelity_res.get("discrepancies", [])
                if discrepancies:
                    with st.expander(f"⚠️ Flagged Scientific Discrepancies ({len(discrepancies)})", expanded=True):
                        st.caption("The following values, p-values, or figure callouts differed between original and enhanced text:")
                        for d in discrepancies:
                            st.markdown(f"- **[{d['severity']}] {d['type']}**: {d['detail']}")
                else:
                    st.success("✅ **100% Scientific Fidelity**: All Figure/Table references, p-values, binding energies, percentages, and active-site residue codes are verified intact!")

            # Visual Inline Diff (Persistent)
            if res_list:
                st.markdown("---")
                diff_idx = 0
                if len(res_list) > 1:
                    chunk_labels = [f"Chunk {i+1} ({r.get('section_type', 'general').replace('_', ' ').title()}) — {r.get('word_count', 0)} words" for i, r in enumerate(res_list)]
                    sel_chunk_label = st.selectbox("Select Chunk to Inspect Visual Inline Diff:", options=chunk_labels, key="t1_chunk_diff_sel")
                    diff_idx = chunk_labels.index(sel_chunk_label)

                with st.expander("🔍 Visual Inline Word Diff (Red: Removed AI / Green: Enhanced Replacement)", expanded=True):
                    st.caption(f"Word-by-word visual difference for Chunk {diff_idx + 1}: red strikethrough denotes deleted robotic patterns; green denotes enhanced scholarly prose:")
                    st.markdown(generate_inline_diff_html(res_list[diff_idx]["original_text"], res_list[diff_idx]["humanized_text"]), unsafe_allow_html=True)

                # Before / After side-by-side comparison
                st.markdown(f"#### Before / After Text Blocks (Chunk {diff_idx + 1})")
                cmp_left, cmp_right = st.columns(2)
                with cmp_left:
                    st.markdown("**Original:**")
                    st.text_area("Original Text", value=res_list[diff_idx]["original_text"], height=230, key="cmp_orig", disabled=True, label_visibility="collapsed")
                with cmp_right:
                    st.markdown("**Enhanced:**")
                    st.text_area("Enhanced Text", value=res_list[diff_idx]["humanized_text"], height=230, key="cmp_human", disabled=True, label_visibility="collapsed")

            # Persistent Downloads
            st.markdown("---")
            st.markdown("#### 📥 Download Enhanced Document")
            st.caption("✅ Downloads preserved — you can download multiple formats sequentially without losing your view:")
            dl_col1, dl_col2, dl_col3, dl_col4 = st.columns(4)
            with dl_col1:
                st.download_button(
                    "📄 Download .docx",
                    data=t1_res["docx_bytes"],
                    file_name=f"Enhanced_{t1_res['base_name']}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_t1_docx"
                )
            with dl_col2:
                st.download_button(
                    "📕 Download .pdf",
                    data=t1_res["pdf_bytes"],
                    file_name=f"Enhanced_{t1_res['base_name']}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_t1_pdf"
                )
            with dl_col3:
                st.download_button(
                    "📝 Download .md",
                    data=t1_res["md_bytes"],
                    file_name=f"{t1_res['base_name']}_enhanced.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_t1_md"
                )
            with dl_col4:
                st.download_button(
                    "📃 Download .txt",
                    data=t1_res["txt_bytes"],
                    file_name=f"{t1_res['base_name']}_enhanced.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_t1_txt"
                )
            if st.button("🔄 Reset / Process New Run", key="clear_tab1_enhance"):
                del st.session_state["tab1_enhance_results"]
                st.rerun()


# =========================================================================== #
# TAB 2: Quick Text Rewriter (Paste Text)
# =========================================================================== #

with tab_text:
    st.markdown("### ✍️ Instant Text & Multi-Document Rewriter")
    st.caption("Paste text directly OR upload one or multiple documents (.docx, .pdf, .txt, .md) to rewrite with academic precision.")

    # Multi-file uploader for Tab 2
    t2_files = st.file_uploader(
        "📂 Upload One or Multiple Files to Load or Merge into Rewriter (.docx, .pdf, .pptx, .txt, .md):",
        type=["docx", "pdf", "pptx", "txt", "md"],
        accept_multiple_files=True,
        key="tab2_multi_files",
        help="Upload multiple documents. You can inspect/load individual files or merge them all into the text rewriter."
    )

    if "tab2_text_content" not in st.session_state:
        st.session_state["tab2_text_content"] = ""
    if "text_input" not in st.session_state:
        st.session_state["text_input"] = ""
    if "_last_t2_files_sig" not in st.session_state:
        st.session_state["_last_t2_files_sig"] = None

    loaded_texts = {}
    chosen_file = None

    if t2_files:
        current_sig = tuple((f.name, f.size) for f in t2_files)
        for f in t2_files:
            fn = f.name.lower()
            if fn.endswith(".docx"):
                loaded_texts[f.name] = extract_text_from_docx_bytes(f.getvalue())
            elif fn.endswith(".pdf"):
                loaded_texts[f.name] = extract_text_from_pdf(f.getvalue())
            elif fn.endswith(".pptx"):
                loaded_texts[f.name] = extract_text_from_pptx(f.getvalue())
            else:
                loaded_texts[f.name] = f.getvalue().decode("utf-8", errors="ignore")

        # Automatically populate editor text upon new file upload
        if st.session_state["_last_t2_files_sig"] != current_sig:
            st.session_state["_last_t2_files_sig"] = current_sig
            first_key = list(loaded_texts.keys())[0]
            st.session_state["tab2_text_content"] = loaded_texts[first_key]
            st.session_state["text_input"] = loaded_texts[first_key]
            st.rerun()

        c_up1, c_up2, c_up3 = st.columns([2, 1, 1])
        with c_up1:
            chosen_file = st.selectbox("Select File from Uploads:", options=list(loaded_texts.keys()), key="t2_file_choice")
        with c_up2:
            if st.button("📥 Load Selected File", use_container_width=True, key="btn_load_single_t2"):
                st.session_state["tab2_text_content"] = loaded_texts[chosen_file]
                st.session_state["text_input"] = loaded_texts[chosen_file]
                st.rerun()
        with c_up3:
            if st.button("📑 Merge All Files", use_container_width=True, key="btn_merge_all_t2"):
                merged = "\n\n".join([f"# Section: {k}\n\n{v}" for k, v in loaded_texts.items()])
                st.session_state["tab2_text_content"] = merged
                st.session_state["text_input"] = merged
                st.rerun()

        if chosen_file and chosen_file in loaded_texts:
            f_words = len(loaded_texts[chosen_file].split())
            st.success(f"✅ Loaded **{f_words:,} words** from `{chosen_file}` into the editor below.")
    else:
        st.session_state["_last_t2_files_sig"] = None

    input_text = st.text_area(
        "Paste or edit your text here:",
        value=st.session_state.get("text_input", st.session_state.get("tab2_text_content", "")),
        height=250,
        placeholder="Paste the text you want to humanize, improve academically, or restructure...",
        key="text_input",
    )
    st.session_state["tab2_text_content"] = input_text

    # Section type selector for text mode
    text_col1, text_col2 = st.columns(2)
    with text_col1:
        section_type = st.selectbox(
            "Section Type (for academic rules)",
            options=["general", "abstract", "introduction", "literature_review", "methodology", "results", "discussion", "conclusion"],
            format_func=lambda x: x.replace("_", " ").title(),
            index=0,
            key="text_section",
        )
    with text_col2:
        st.markdown(f"**Mode:** {'🔴 Aggressive' if aggressive_mode else '🟢 Balanced'}")
        active_opts = [k.replace('_', ' ').title() for k, v in user_options.items() if v]
        st.markdown(f"**Active:** {', '.join(active_opts) if active_opts else 'None'}")

    # Specialized IMRaD banner for Abstracts (PMC6398294 Guidelines)
    if section_type == "abstract":
        st.info("""
**📑 Academic Abstract Protocol (IMRaD & PMC6398294 Guidelines):**
* **The 5-Part Flow:** Background (1–2 sent) ➔ Aim / Objective (1 sent) ➔ Methods (2–3 sent) ➔ Results (2–3 empirical findings with data) ➔ Conclusion / Significance (1–2 sent).
* **Tenses:** Present tense for background truths & overarching conclusions; Past tense for completed study methods & results; STRICTLY NO future tense.
* **Strictly Self-Contained:** NEVER include outside citations (no `[1]`, no `Smith (2020)`), never cite figures or tables from the main text, and conclude with 3–6 MeSH keywords.
""")
    elif section_type == "conclusion":
        st.info("""
**🎯 Academic Conclusion Protocol:**
* **The 4-Step Flow:**
  1. **Restate the Main Idea:** Paraphrase the central thesis using fresh, elevated vocabulary — never copy verbatim from the introduction.
  2. **Synthesize Key Points:** Connect core findings together to demonstrate cohesive evidence rather than producing an uninspired serial list.
  3. **Answer the "So What?":** Situate the study within the broader scientific context and articulate why the outcomes matter.
  4. **Leave a Forward-Looking Final Thought:** End with high-impact future research trajectories (linked to study constraints) or an authoritative takeaway.
* **Scientific Standard:** Summarize and synthesize; never introduce unanalyzed raw data or unproven theories; eliminate clichés like *"In conclusion"* or *"To sum up"*.
""")

    # Target Word Count & Length Control
    st.markdown("##### 🎯 Target Word Count & Length Strategy")
    st.caption("Control section length precisely, prevent shrinkage when removing AI filler, or set exact word bounds.")

    if section_type == "abstract":
        strat_opts = [
            "Preserve Original Length (~100% of input, prevents shrinkage)",
            "Journal Abstract Preset (250–300 words)",
            "Short Abstract Preset (150–250 words)",
            "Micro Summary (100–150 words)",
            "Custom Target Range (specify Min - Max)",
            "Tighten & Condense (-25% words)",
            "Elaborate & Deepen (+25% words)",
        ]
        def_idx = 1
    elif section_type == "conclusion":
        strat_opts = [
            "Preserve Original Length (~100% of input, default)",
            "Concise Journal Conclusion (200–350 words)",
            "Comprehensive Thesis / Chapter Conclusion (450–700 words)",
            "Executive Takeaway (120–200 words)",
            "Custom Target Range (specify Min - Max)",
            "Tighten & Condense (-25% words)",
            "Elaborate & Deepen (+25% words)",
        ]
        def_idx = 0
    else:
        strat_opts = [
            "Preserve Original Length (~100% of input, default)",
            "Short Academic Section (300–500 words)",
            "Standard Academic Section (600–1000 words)",
            "Comprehensive Section (1200–1800 words)",
            "Custom Target Range (specify Min - Max)",
            "Tighten & Condense (-25% words)",
            "Elaborate & Deepen (+25% words)",
        ]
        def_idx = 0

    len_col_left, len_col_right = st.columns([1.6, 1])
    with len_col_left:
        length_choice = st.radio(
            "Length Strategy:",
            options=strat_opts,
            index=def_idx,
            key=f"tab2_len_choice_{section_type}",
        )

    target_min = None
    target_max = None
    if "Preserve" in length_choice:
        length_mode = "preserve"
    elif "Journal Abstract" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 250, 300
    elif "Short Abstract" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 150, 250
    elif "Micro Summary" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 100, 150
    elif "Concise Journal Conclusion" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 200, 350
    elif "Comprehensive Thesis / Chapter Conclusion" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 450, 700
    elif "Executive Takeaway" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 120, 200
    elif "Short Academic Section" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 300, 500
    elif "Standard Academic Section" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 600, 1000
    elif "Comprehensive Section" in length_choice:
        length_mode = "target_range"
        target_min, target_max = 1200, 1800
    elif "Custom Target Range" in length_choice:
        length_mode = "target_range"
        with len_col_right:
            target_min = st.number_input("Minimum Words", min_value=30, max_value=5000, value=300, step=25, key="t2_min")
            target_max = st.number_input("Maximum Words", min_value=int(target_min), max_value=6000, value=max(400, int(target_min) + 50), step=25, key="t2_max")
    elif "Tighten" in length_choice:
        length_mode = "concise"
    elif "Elaborate" in length_choice:
        length_mode = "elaborate"

    if input_text.strip():
        # Pre-scan
        st.markdown("#### Input AI Marker Scan")
        text_pre_audit = render_ai_audit(input_text, label="Input")

        word_count = len(input_text.split())
        st.caption(f"Input Word Count: **{word_count} words**")

    if st.button("🚀 Rewrite Text", type="primary", use_container_width=True, key="text_process"):
        # Auto-fallback: if text area is empty but file was uploaded, use uploaded file text
        current_input = st.session_state.get("text_input", "").strip() or input_text.strip()
        if not current_input and t2_files and loaded_texts:
            active_name = chosen_file or list(loaded_texts.keys())[0]
            current_input = loaded_texts.get(active_name, "").strip()
            st.session_state["text_input"] = current_input
            st.session_state["tab2_text_content"] = current_input
            input_text = current_input

        if not current_input:
            st.warning("⚠️ Please paste some text or upload a document to rewrite.")
        elif not api_key:
            st.error("⚠️ Please enter your API Key in the sidebar.")
        elif not any(user_options.values()):
            st.error("⚠️ Please select at least one processing option in the sidebar.")
        else:
            try:
                engine = get_engine()

                with st.spinner("Processing text with word count boundaries..."):
                    humanized = engine.rewrite_text(
                        text=current_input,
                        section_type=section_type,
                        aggressive=aggressive_mode,
                        options=user_options,
                        length_mode=length_mode,
                        target_min=target_min,
                        target_max=target_max,
                        work_mode=opt_work_mode,
                        english_tone=opt_english_tone,
                        strict_mode=opt_strict_mode,
                    )

                text_post_audit = render_ai_audit(humanized, label="Output")
                text_pre = analyze_ai_patterns(current_input)
                delta = text_pre["score"] - text_post_audit["score"]

                in_w = len(current_input.split())
                out_w = len(humanized.split())

                text_ngram = calculate_ngram_similarity(current_input, humanized, n=4)

                # Prepare and cache all download formats
                docx_bytes_t2 = text_to_docx_bytes(
                    humanized,
                    typography_preset=opt_font_family,
                    margins_inches=1.0 if opt_margins else 0.75,
                    line_spacing=opt_line_spacing,
                    include_title_page=opt_title_page,
                    title_page_data=title_page_dict,
                    include_toc=opt_toc,
                    alignment=clean_alignment,
                )
                pdf_bytes_t2 = text_to_pdf_bytes(
                    humanized,
                    title=title_page_dict.get("title", f"Enhanced {section_type.title()}") if opt_title_page else f"Enhanced {section_type.title()}",
                    include_title_page=opt_title_page,
                    title_page_data=title_page_dict,
                    include_toc=opt_toc,
                    alignment=clean_alignment,
                )
                import re as _re
                plain_text_t2 = _re.sub(r"^#{1,6}\s+", "", humanized, flags=_re.MULTILINE)

                st.session_state["tab2_results"] = {
                    "humanized": humanized,
                    "input_text": current_input,
                    "section_type": section_type,
                    "delta": delta,
                    "pre_score": text_pre["score"],
                    "post_score": text_post_audit["score"],
                    "in_w": in_w,
                    "out_w": out_w,
                    "target_min": target_min,
                    "target_max": target_max,
                    "length_choice": length_choice,
                    "text_ngram": text_ngram,
                    "docx_bytes": docx_bytes_t2,
                    "pdf_bytes": pdf_bytes_t2,
                    "md_bytes": humanized.encode("utf-8"),
                    "txt_bytes": plain_text_t2.encode("utf-8"),
                }
                st.rerun()

            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")

    # Persistent Display of Tab 2 Results & Downloads (Never disappears upon clicking download or interaction)
    if "tab2_results" in st.session_state:
        t2_res = st.session_state["tab2_results"]
        st.markdown("---")
        st.markdown("### 📊 Enhancement Results")

        # Score delta banner
        if t2_res.get("delta", 0) > 0:
            st.success(f"📉 AI Score reduced by **{t2_res['delta']:.1f}** points ({t2_res['pre_score']:.1f} → {t2_res['post_score']:.1f})")

        # Word Count Verification
        st.markdown("#### 📊 Word Count & Target Verification")
        wc1, wc2, wc3 = st.columns(3)
        wc1.metric("Input Words", t2_res["in_w"])
        wc2.metric("Enhanced Words", t2_res["out_w"], delta=t2_res["out_w"] - t2_res["in_w"])
        if t2_res.get("target_min") and t2_res.get("target_max"):
            is_within = t2_res["target_min"] <= t2_res["out_w"] <= t2_res["target_max"]
            wc3.metric(f"Target Window ({t2_res['target_min']}–{t2_res['target_max']})", "✅ Within Range" if is_within else "⚠️ Outside Range")
        else:
            wc3.metric("Length Strategy", t2_res.get("length_choice", "").split(" (")[0])

        # N-gram similarity & Originality scan
        t2_ngram = t2_res.get("text_ngram", {})
        if t2_ngram:
            st.markdown("#### 🛡️ Plagiarism & N-Gram Overlap Scan")
            ng_a, ng_b, ng_c = st.columns(3)
            ng_a.metric("Originality Score", f"🛡️ {t2_ngram.get('originality_score', 100)}%")
            ng_b.metric("4-Gram Overlap", f"{t2_ngram.get('overlap_percentage', 0)}%")
            ng_c.metric("Plagiarism Risk", t2_ngram.get("risk_level", "Low"))
            if t2_ngram.get("matching_sequences"):
                with st.expander(f"⚠️ Matching 4-Gram Sequences ({len(t2_ngram['matching_sequences'])})", expanded=False):
                    st.caption("These exact 4-word sequences match the source text:")
                    st.write(", ".join([f"`{seq}`" for seq in t2_ngram["matching_sequences"][:25]]))

        # Visual Inline Diff (Persistent)
        st.markdown("---")
        with st.expander("🔍 Visual Inline Word Diff (Red: Removed AI / Green: Enhanced Replacement)", expanded=True):
            st.caption("Word-by-word visual difference: red strikethrough denotes deleted robotic patterns; green denotes enhanced scholarly prose:")
            st.markdown(generate_inline_diff_html(t2_res["input_text"], t2_res["humanized"]), unsafe_allow_html=True)

        # Side-by-side Before / After
        st.markdown("#### Before / After Text Blocks")
        left_col, right_col = st.columns(2)
        with left_col:
            st.markdown("**Original:**")
            st.text_area("Original Input", value=t2_res["input_text"], height=250, key="txt_orig", disabled=True, label_visibility="collapsed")
        with right_col:
            st.markdown("**Enhanced:**")
            st.text_area("Enhanced Output", value=t2_res["humanized"], height=250, key="txt_result", disabled=True, label_visibility="collapsed")

        # Persistent Downloads
        st.markdown("---")
        st.markdown("#### 📥 Download Enhanced Result")
        st.caption("✅ Downloads preserved — download Word, PDF, Markdown, or TXT without losing your screen state:")
        dl_a, dl_b, dl_c, dl_d = st.columns(4)
        with dl_a:
            st.download_button(
                "📄 Download .docx",
                data=t2_res["docx_bytes"],
                file_name=f"enhanced_{t2_res['section_type']}.docx",
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                use_container_width=True,
                key="dl_t2_docx"
            )
        with dl_b:
            st.download_button(
                "📕 Download .pdf",
                data=t2_res["pdf_bytes"],
                file_name=f"enhanced_{t2_res['section_type']}.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="dl_t2_pdf"
            )
        with dl_c:
            st.download_button(
                "📝 Download .md",
                data=t2_res["md_bytes"],
                file_name=f"enhanced_{t2_res['section_type']}.md",
                mime="text/markdown",
                use_container_width=True,
                key="dl_t2_md"
            )
        with dl_d:
            st.download_button(
                "📃 Download .txt",
                data=t2_res["txt_bytes"],
                file_name=f"enhanced_{t2_res['section_type']}.txt",
                mime="text/plain",
                use_container_width=True,
                key="dl_t2_txt"
            )
        if st.button("🔄 Clear Results / New Text", key="clear_tab2_results"):
            del st.session_state["tab2_results"]
            st.rerun()


# =========================================================================== #
# TAB 3: Plagiarism & Paraphrasing Studio
# =========================================================================== #

with tab_plagiarism:
    st.markdown("### 🛡️ Scientific Paraphrasing & Plagiarism Reduction Studio")
    st.caption("Defeat string-matching algorithms (Turnitin, iThenticate, Grammarly) through deep N-Gram disruption, syntactic flipping, and semantic reconstruction.")

    plag_mode = st.radio(
        "Select Workflow:",
        options=["⚡ Deep Paraphraser (N-Gram Breaking)", "🔍 Similarity & Overlap Inspector (Draft vs. Source)", "📚 Scientific Paraphrasing Masterclass"],
        horizontal=True,
    )

    if plag_mode == "⚡ Deep Paraphraser (N-Gram Breaking)":
        st.markdown("#### Input Source Passage to Paraphrase")
        plag_input_type = st.radio(
            "Input Method:",
            options=["📝 Direct Text Paste", "📁 Upload Document (.docx, .pdf, .pptx, .txt, .md)"],
            horizontal=True,
            key="plag_input_type"
        )

        plag_input = ""
        if plag_input_type == "📁 Upload Document (.docx, .pdf, .pptx, .txt, .md)":
            p_file = st.file_uploader(
                "Upload document to paraphrase:",
                type=["docx", "pdf", "pptx", "txt", "md"],
                key="plag_file_uploader",
                help="Extracts text directly from your Word, PDF, PowerPoint, Markdown, or Text file."
            )
            if p_file is not None:
                if p_file.name.lower().endswith(".pdf"):
                    plag_input = extract_text_from_pdf(p_file.getvalue())
                elif p_file.name.lower().endswith(".pptx"):
                    plag_input = extract_text_from_pptx(p_file.getvalue())
                elif p_file.name.lower().endswith((".txt", ".md")):
                    plag_input = p_file.getvalue().decode("utf-8", errors="ignore")
                else:  # .docx
                    import docx as _docx
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
                        tmp.write(p_file.getvalue())
                        tmp_name = tmp.name
                    try:
                        d = _docx.Document(tmp_name)
                        plag_input = "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
                    finally:
                        try:
                            os.remove(tmp_name)
                        except OSError:
                            pass
                st.success(f"📁 Extracted **{len(plag_input.split()):,} words** from `{p_file.name}`")
                with st.expander("Preview Extracted Source Text", expanded=False):
                    st.text_area("Extracted Source Text", value=plag_input, height=180, disabled=True, key="plag_preview", label_visibility="collapsed")
        else:
            plag_input = st.text_area(
                "Paste original text:",
                height=200,
                placeholder="Paste text with high similarity or duplicate phrasing that needs to be rewritten from scratch...",
                key="plag_rewrite_input"
            )
        
        c_sec, c_opt = st.columns([1, 1])
        with c_sec:
            p_sec_type = st.selectbox(
                "Academic Section:",
                options=["general", "introduction", "literature_review", "methodology", "results", "discussion", "conclusion"],
                format_func=lambda x: x.replace("_", " ").title(),
                key="plag_sec_type"
            )
        with c_opt:
            p_aggressive = st.checkbox("🔥 Ultra-Low Similarity Mode (Deep Token & Clause Inversion)", value=True)

        if st.button("🚀 Paraphrase & Eliminate N-Grams", type="primary", use_container_width=True, key="plag_btn"):
            if not plag_input.strip():
                st.warning("⚠️ Please provide or paste text to paraphrase.")
            elif not api_key:
                st.error("⚠️ Please enter your API Key in the sidebar.")
            else:
                try:
                    engine = get_engine()
                    with st.spinner("Applying 'Read and Shield' protocol & syntactic flipping..."):
                        paraphrased = engine.rewrite_text(
                            text=plag_input,
                            section_type=p_sec_type,
                            aggressive=p_aggressive,
                            options=user_options,
                            work_mode=opt_work_mode,
                            english_tone=opt_english_tone,
                            strict_mode=opt_strict_mode,
                        )

                    # N-gram analysis
                    sim = calculate_ngram_similarity(plag_input, paraphrased, n=4)

                    # Prepare and cache all download formats
                    docx_bytes_t3 = text_to_docx_bytes(
                        paraphrased,
                        typography_preset=opt_font_family,
                        margins_inches=1.0 if opt_margins else 0.75,
                        line_spacing=opt_line_spacing,
                        include_title_page=opt_title_page,
                        title_page_data=title_page_dict,
                        include_toc=opt_toc,
                        alignment=clean_alignment,
                    )
                    pdf_bytes_t3 = text_to_pdf_bytes(
                        paraphrased,
                        title=title_page_dict.get("title", "Paraphrased Academic Document") if opt_title_page else "Paraphrased Academic Document",
                        include_title_page=opt_title_page,
                        title_page_data=title_page_dict,
                        include_toc=opt_toc,
                        alignment=clean_alignment,
                    )
                    import re as _re
                    plain_text_t3 = _re.sub(r"^#{1,6}\s+", "", paraphrased, flags=_re.MULTILINE)

                    st.session_state["tab3_results"] = {
                        "paraphrased": paraphrased,
                        "plag_input": plag_input,
                        "sim": sim,
                        "docx_bytes": docx_bytes_t3,
                        "pdf_bytes": pdf_bytes_t3,
                        "md_bytes": paraphrased.encode("utf-8"),
                        "txt_bytes": plain_text_t3.encode("utf-8"),
                    }
                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Paraphrasing failed: {str(e)}")

        # Persistent Display of Tab 3 Results & Downloads (Never disappears upon clicking download or interaction)
        if "tab3_results" in st.session_state:
            t3_res = st.session_state["tab3_results"]
            sim = t3_res.get("sim", {})

            st.markdown("---")
            st.markdown("### 📊 Paraphrasing & Originality Results")
            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("Originality Score", f"🛡️ {sim.get('originality_score', 100)}%")
            col_m2.metric("4-Gram Overlap", f"{sim.get('overlap_percentage', 0)}%")
            col_m3.metric("Similarity Risk", sim.get("risk_level", "Low"))

            if sim.get("matching_sequences"):
                with st.expander(f"⚠️ Identical 4-Gram Sequences Found ({len(sim['matching_sequences'])})", expanded=False):
                    st.caption("These sequences match verbatim:")
                    st.write(", ".join([f"`{s}`" for s in sim["matching_sequences"][:25]]))
            else:
                st.success("✅ Zero 4-Gram matches detected! The text structure has been completely transformed.")

            # Visual Inline Diff (Persistent)
            st.markdown("---")
            with st.expander("🔍 Visual Inline Word Diff (Red: Source Phrasing / Green: Unique Paraphrase)", expanded=True):
                st.caption("Word-by-word visual difference: red strikethrough denotes source wording; green denotes unique paraphrased structure:")
                st.markdown(generate_inline_diff_html(t3_res["plag_input"], t3_res["paraphrased"]), unsafe_allow_html=True)

            # Side-by-side Before / After
            st.markdown("#### Before / After Text Blocks")
            p_col1, p_col2 = st.columns(2)
            with p_col1:
                st.markdown("**Original Source:**")
                st.text_area("Original Source", value=t3_res["plag_input"], height=280, key="plag_res_orig", disabled=True, label_visibility="collapsed")
            with p_col2:
                st.markdown("**Original Paraphrased Output:**")
                st.text_area("Paraphrased Output", value=t3_res["paraphrased"], height=280, key="plag_res_out", disabled=True, label_visibility="collapsed")

            # Persistent Downloads
            st.markdown("---")
            st.markdown("#### 📥 Download Paraphrased Result")
            st.caption("✅ Downloads preserved — download Word, PDF, Markdown, or TXT without losing your screen state:")
            pd_1, pd_2, pd_3, pd_4 = st.columns(4)
            with pd_1:
                st.download_button(
                    "📄 Download .docx",
                    data=t3_res["docx_bytes"],
                    file_name="paraphrased.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_t3_docx"
                )
            with pd_2:
                st.download_button(
                    "📕 Download .pdf",
                    data=t3_res["pdf_bytes"],
                    file_name="paraphrased.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_t3_pdf"
                )
            with pd_3:
                st.download_button(
                    "📝 Download .md",
                    data=t3_res["md_bytes"],
                    file_name="paraphrased.md",
                    mime="text/markdown",
                    use_container_width=True,
                    key="dl_t3_md"
                )
            with pd_4:
                st.download_button(
                    "📃 Download .txt",
                    data=t3_res["txt_bytes"],
                    file_name="paraphrased.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_t3_txt"
                )
            if st.button("🔄 Clear Results / New Paraphrase", key="clear_tab3_results"):
                del st.session_state["tab3_results"]
                st.rerun()

    elif plag_mode == "🔍 Similarity & Overlap Inspector (Draft vs. Source)":
        st.markdown("#### 🔍 Multi-Source Overlap Inspector & Citation Authenticator")
        st.caption("Cross-check your draft against multiple reference articles, books, or papers (.pdf, .docx, .pptx, .txt) simultaneously. Detect verbatim n-grams per source, assess originality, and auto-generate standard citations.")

        col_src_panel, col_dft_panel = st.columns(2)

        # --- Left Panel: Reference Sources (Multi-File or Paste) ---
        with col_src_panel:
            st.markdown("##### 📚 Reference Sources (Corpus)")
            src_input_method = st.radio(
                "Source Input Method:",
                options=["📂 Upload Reference Articles / Papers", "📝 Paste Source Text"],
                horizontal=True,
                key="src_input_method"
            )

            # Dictionary of {filename/label: text}
            source_docs = {}

            if src_input_method == "📂 Upload Reference Articles / Papers":
                src_files = st.file_uploader(
                    "Upload Reference Files (.pdf, .docx, .pptx, .txt, .md):",
                    type=["pdf", "docx", "pptx", "txt", "md"],
                    accept_multiple_files=True,
                    key="comp_src_files",
                    help="Upload multiple research papers, textbooks, or reference articles."
                )
                if src_files:
                    for sf in src_files:
                        sfn = sf.name.lower()
                        if sfn.endswith(".pdf"):
                            source_docs[sf.name] = extract_text_from_pdf(sf.getvalue())
                        elif sfn.endswith(".pptx"):
                            source_docs[sf.name] = extract_text_from_pptx(sf.getvalue())
                        elif sfn.endswith(".docx"):
                            source_docs[sf.name] = extract_text_from_docx_bytes(sf.getvalue())
                        else:
                            source_docs[sf.name] = sf.getvalue().decode("utf-8", errors="ignore")
                    
                    total_src_words = sum(len(txt.split()) for txt in source_docs.values())
                    st.success(f"📚 Loaded **{len(source_docs)} Reference Documents** ({total_src_words:,} total words)")
                    with st.expander(f"View Loaded Reference Files ({len(source_docs)})", expanded=False):
                        for s_name, s_txt in source_docs.items():
                            st.markdown(f"**📄 `{s_name}`** ({len(s_txt.split()):,} words)")
                else:
                    st.info("💡 Upload one or multiple reference papers (e.g. PDFs, Word docs) to build your reference corpus.")
            else:
                pasted_source = st.text_area(
                    "Paste original source text:",
                    height=240,
                    placeholder="Paste the reference paragraph, journal excerpt, or book chapter...",
                    key="comp_src_pasted"
                )
                if pasted_source.strip():
                    source_docs["Pasted Reference"] = pasted_source.strip()

        # --- Right Panel: Your Draft Manuscript ---
        with col_dft_panel:
            st.markdown("##### ✍️ Your Draft Manuscript")
            dft_input_method = st.radio(
                "Draft Input Method:",
                options=["📝 Paste Draft Text", "📁 Upload Draft File"],
                horizontal=True,
                key="dft_input_method"
            )

            draft_content = ""
            if dft_input_method == "📁 Upload Draft File":
                dft_file = st.file_uploader(
                    "Upload your draft (.docx, .pdf, .pptx, .txt, .md):",
                    type=["docx", "pdf", "pptx", "txt", "md"],
                    key="comp_dft_file",
                    help="Upload your thesis chapter, assignment, or paper draft."
                )
                if dft_file is not None:
                    dfn = dft_file.name.lower()
                    if dfn.endswith(".pdf"):
                        draft_content = extract_text_from_pdf(dft_file.getvalue())
                    elif dfn.endswith(".pptx"):
                        draft_content = extract_text_from_pptx(dft_file.getvalue())
                    elif dfn.endswith(".docx"):
                        draft_content = extract_text_from_docx_bytes(dft_file.getvalue())
                    else:
                        draft_content = dft_file.getvalue().decode("utf-8", errors="ignore")
                    st.success(f"✍️ Loaded draft `{dft_file.name}` ({len(draft_content.split()):,} words)")
                    with st.expander("Preview Draft Content", expanded=False):
                        st.text_area("Preview Draft Content", value=draft_content, height=160, disabled=True, key="dft_preview_box", label_visibility="collapsed")
            else:
                draft_content = st.text_area(
                    "Paste your draft here:",
                    height=240,
                    placeholder="Paste the draft text or section you wrote to authenticate and check for similarity...",
                    key="comp_dft_pasted"
                )

        # --- Analysis Settings ---
        st.markdown("---")
        c_set1, c_set2 = st.columns([1, 1])
        with c_set1:
            n_val = st.slider(
                "N-Gram String Matching Window:",
                min_value=3,
                max_value=6,
                value=4,
                help="Turnitin and iThenticate flag continuous runs of 4 to 6 words. 4 is the universal benchmark."
            )
        with c_set2:
            cite_style = st.selectbox(
                "Citation & Reference Style:",
                options=["APA 7th Edition", "IEEE / Numbered [1]", "Harvard"],
                key="cite_style_choice"
            )

        if st.button("🔬 Analyze String Overlap & Authenticate Sources", type="primary", use_container_width=True, key="btn_inspect_multi"):
            if not source_docs:
                st.warning("⚠️ Please provide at least one reference source (upload files or paste text).")
            elif not draft_content.strip():
                st.warning("⚠️ Please provide your draft text (upload file or paste text).")
            else:
                pooled_source = "\n\n".join(source_docs.values())
                combined_sim = calculate_ngram_similarity(pooled_source, draft_content, n=n_val)

                per_source_results = []
                for idx, (s_name, s_txt) in enumerate(source_docs.items(), start=1):
                    s_sim = calculate_ngram_similarity(s_txt, draft_content, n=n_val)
                    cite_data = generate_source_citation_entry(s_name, s_txt, idx, style=cite_style)
                    per_source_results.append({
                        "name": s_name,
                        "words": len(s_txt.split()),
                        "sim": s_sim,
                        "cite": cite_data,
                        "idx": idx,
                    })

                per_source_results.sort(key=lambda x: x["sim"]["overlap_percentage"], reverse=True)

                st.markdown("---")
                st.markdown("### 📊 Cross-Source Plagiarism & Authentication Report")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Overall Originality", f"🛡️ {combined_sim['originality_score']}%")
                m2.metric(f"Total {n_val}-Gram Overlap", f"{combined_sim['overlap_percentage']}%")
                m3.metric("Reference Sources Checked", f"{len(source_docs)} Files")
                m4.metric("Similarity Risk Level", combined_sim["risk_level"])

                st.markdown("#### 📑 Source-by-Source Similarity Breakdown")
                breakdown_rows = []
                for r in per_source_results:
                    s_ov = r["sim"]["overlap_percentage"]
                    s_matches = len(r["sim"]["matching_sequences"])
                    status_badge = "🟢 Clean (<1%)" if s_ov < 1.0 else ("🟡 Moderate (1-5%)" if s_ov < 5.0 else "🔴 High Similarity (>5%)")
                    breakdown_rows.append({
                        "Source File": f"`{r['name']}`",
                        "Source Words": f"{r['words']:,}",
                        f"{n_val}-Gram Overlap": f"{s_ov}%",
                        "Verbatim Runs": s_matches,
                        "Suggested In-Text Citation": f"`{r['cite']['in_text']}`",
                        "Risk Status": status_badge,
                    })
                st.table(breakdown_rows)

                has_flags = any(len(r["sim"]["matching_sequences"]) > 0 for r in per_source_results)
                if has_flags:
                    st.markdown("#### ⚠️ Specific Overlapping Phrases Identified by Source File")
                    for r in per_source_results:
                        if r["sim"]["matching_sequences"]:
                            with st.expander(f"🚩 Overlaps with `{r['name']}` ({len(r['sim']['matching_sequences'])} matching {n_val}-grams)", expanded=True):
                                st.caption(f"Suggested In-Text Citation: `{r['cite']['in_text']}` — Insert this citation or paraphrase:")
                                st.write(", ".join([f"`{seq}`" for seq in r["sim"]["matching_sequences"][:30]]))
                else:
                    st.success(f"🎉 Excellent! Zero verbatim {n_val}-word sequences found across any of your {len(source_docs)} reference documents.")

                st.markdown("---")
                st.markdown(f"#### 📚 Auto-Generated References Section ({cite_style})")
                st.caption("Copy these citations or download your draft with this reference list appended:")
                
                ref_list_lines = [r["cite"]["bib_entry"] for r in per_source_results]
                formatted_refs_text = "## References\n\n" + "\n\n".join(ref_list_lines)
                st.text_area("Formatted Bibliography:", value=formatted_refs_text, height=170, key="ref_list_display")

                authenticated_draft = f"{draft_content.strip()}\n\n---\n\n{formatted_refs_text}"
                
                doc_bytes_auth = text_to_docx_bytes(
                    authenticated_draft,
                    typography_preset=opt_font_family,
                    margins_inches=1.0 if opt_margins else 0.75,
                    line_spacing=opt_line_spacing,
                    include_title_page=opt_title_page,
                    title_page_data=title_page_dict,
                    include_toc=False,
                    alignment=clean_alignment,
                )
                pdf_bytes_auth = text_to_pdf_bytes(
                    authenticated_draft,
                    title=title_page_dict.get("title", "Authenticated Academic Draft") if opt_title_page else "Authenticated Academic Draft",
                    include_title_page=opt_title_page,
                    title_page_data=title_page_dict,
                    include_toc=False,
                    alignment=clean_alignment,
                )

                st.session_state["tab3_auth_results"] = {
                    "draft_with_refs": authenticated_draft,
                    "docx_bytes": doc_bytes_auth,
                    "pdf_bytes": pdf_bytes_auth,
                    "txt_bytes": authenticated_draft.encode("utf-8"),
                }

        # Persistent download of Authenticated Draft with References
        if "tab3_auth_results" in st.session_state:
            auth_r = st.session_state["tab3_auth_results"]
            st.markdown("---")
            st.markdown("#### 📥 Download Authenticated Draft (with References Appended)")
            st.caption("✅ Downloads preserved — download your draft containing all cross-checked citations and references:")
            ad_1, ad_2, ad_3 = st.columns(3)
            with ad_1:
                st.download_button(
                    "📄 Download Authenticated .docx",
                    data=auth_r["docx_bytes"],
                    file_name="Authenticated_Draft_with_References.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_auth_docx"
                )
            with ad_2:
                st.download_button(
                    "📕 Download Authenticated .pdf",
                    data=auth_r["pdf_bytes"],
                    file_name="Authenticated_Draft_with_References.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                    key="dl_auth_pdf"
                )
            with ad_3:
                st.download_button(
                    "📃 Download Authenticated .txt",
                    data=auth_r["txt_bytes"],
                    file_name="Authenticated_Draft_with_References.txt",
                    mime="text/plain",
                    use_container_width=True,
                    key="dl_auth_txt"
                )
            if st.button("🔄 Reset Inspection", key="clear_auth_results"):
                del st.session_state["tab3_auth_results"]
                st.rerun()

    elif plag_mode == "📚 Scientific Paraphrasing Masterclass":
        st.markdown("#### Scientific Paraphrasing & Academic Ethics Protocol")
        st.markdown("""
To eliminate plagiarism while maintaining academic integrity, apply the four scientific pillars:

##### 1. The "Read and Shield" Technique
- **Read** the source passage multiple times until you fully master the underlying empirical argument.
- **Shield** (hide) the source completely.
- **Write** the idea from memory as if explaining it to a peer.
- **Check** your draft against the original to ensure no 4-word identical sequences remain.

##### 2. Syntactic Flipping (Clause Inversion)
Invert cause-and-effect clauses or alternate active/passive voice:
- *Original (Cause $\\rightarrow$ Effect):* "Because the global temperature is rising, polar ice caps are melting at an alarming rate."
- *Effective Paraphrase (Effect $\\rightarrow$ Cause):* "Accelerated polar ice depletion represents a direct, measurable consequence of escalating planetary temperatures."

##### 3. Part-of-Speech Transformation (Nominalization)
Convert verbs to nouns or adjectives to adverbs:
- *Original:* "The team successfully implemented the new diagnostic protocol."
- *Effective Paraphrase:* "Successful implementation of the revised diagnostic protocol was achieved through clinical team oversight."

##### 4. The Inviolable Citation Rule
Even when you achieve 100% originality score and zero N-gram overlap, **you must still provide an in-text citation** (e.g. `(Smith et al., 2023)` or `[1]`) to credit the original intellectual contribution.
""")


# =========================================================================== #
# TAB 4: AI Marker & Watermark Auditor (Diagnostic Only)
# =========================================================================== #


with tab_audit:
    st.markdown("### Inspect text for AI writing markers")
    st.caption("Paste any text to get a detailed diagnostic — no rewriting, no API key needed.")

    audit_text = st.text_area(
        "Paste text to audit:",
        height=250,
        placeholder="Paste text here to check for AI writing patterns, banned vocabulary, structural markers, and tone issues...",
        key="audit_input",
    )

    c_btn1, c_btn2 = st.columns(2)
    with c_btn1:
        run_audit = st.button("🔬 Run AI Diagnostic Audit", type="primary", use_container_width=True, key="audit_run")
    with c_btn2:
        run_sanitize = st.button("✨ 1-Click Anti-AI Sanitizer (Remove Em-Dashes & Watermarks)", use_container_width=True, key="audit_sanitize_btn")

    if run_sanitize:
        if not audit_text.strip():
            st.warning("⚠️ Please paste some text to sanitize.")
        else:
            from core.prompt_rules import sanitize_ai_markers
            sanitized_ver = sanitize_ai_markers(audit_text)
            st.session_state["tab4_sanitized_text"] = sanitized_ver
            st.success("✅ Text sanitized! All em-dashes ('—') converted to standard commas/punctuation and invisible Unicode watermarks removed.")

    if "tab4_sanitized_text" in st.session_state:
        st.markdown("---")
        st.markdown("#### ✨ Sanitized Clean Text (ZeroGPT / Turnitin Cleaned)")
        st.text_area("Cleaned Text:", value=st.session_state["tab4_sanitized_text"], height=200, key="sanitized_display")
        c_cp1, c_cp2 = st.columns(2)
        with c_cp1:
            st.download_button("📄 Download Cleaned .docx", data=text_to_docx_bytes(st.session_state["tab4_sanitized_text"], alignment=clean_alignment), file_name="sanitized_text.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", use_container_width=True)
        with c_cp2:
            st.download_button("📃 Download Cleaned .txt", data=st.session_state["tab4_sanitized_text"].encode("utf-8"), file_name="sanitized_text.txt", mime="text/plain", use_container_width=True)

    if run_audit:
        if not audit_text.strip():
            st.warning("⚠️ Please paste some text to audit.")
        else:
            st.markdown("---")
            st.markdown("#### AI Pattern & Detector Vulnerability Analysis")
            audit_result = render_ai_audit(audit_text)
            render_detailed_audit(audit_result)

            # Em-Dash and Hyphen Footprint Detection (Turnitin & ZeroGPT #1 trigger)
            em_dash_count = audit_text.count("—")
            en_dash_count = audit_text.count("–")
            double_hyphen_count = audit_text.count("--")
            
            st.markdown("#### ⚡ Turnitin & ZeroGPT Structural Footprint Analysis")
            f_col1, f_col2, f_col3 = st.columns(3)
            f_col1.metric("Em-Dashes ('—')", em_dash_count)
            f_col2.metric("En-Dashes ('–')", en_dash_count)
            f_col3.metric("Double Hyphens ('--')", double_hyphen_count)

            if em_dash_count > 0:
                st.error(f"🚨 **High Risk AI Marker Detected:** Found **{em_dash_count} em-dash ('—')** parenthetical(s). Modern AI classifiers (Turnitin, ZeroGPT, GPTZero) heavily penalize em-dashes as synthetic machine phrasing. Click the '✨ 1-Click Anti-AI Sanitizer' button above to convert them automatically.")
            else:
                st.success("✅ Zero em-dashes detected — excellent human punctuation hygiene.")

            # Sentence Burstiness Analysis (Perplexity & Rhythm Variance)
            import statistics
            sents = [s.strip() for s in re.split(r"[.!?]+", audit_text) if s.strip()]
            if len(sents) >= 2:
                lengths = [len(s.split()) for s in sents]
                mean_l = round(statistics.mean(lengths), 1)
                stdev_l = round(statistics.stdev(lengths), 1)
                burst_label = "🟢 High Burstiness (Authentic Human Variety)" if stdev_l >= 7.0 else ("🟡 Moderate Variance" if stdev_l >= 4.0 else "🔴 Low Burstiness (Monotonous AI Pattern)")
                
                b_c1, b_c2, b_c3 = st.columns(3)
                b_c1.metric("Avg Sentence Length", f"{mean_l} words")
                b_c2.metric("Burstiness (Std Dev)", f"{stdev_l}")
                b_c3.metric("Rhythm Profile", burst_label)
                if stdev_l < 4.0:
                    st.warning("⚠️ **Low Burstiness Warning:** Sentences have uniform lengths. AI detectors flag this rhythmic regularity. Human writing requires mixing 5-word statements with 30-word analyses.")

            # Unicode scan
            _, unicode_stats = sanitize_unicode(audit_text)
            if unicode_stats["removed_invisibles"] > 0 or unicode_stats["normalized_spaces"] > 0:
                st.markdown("#### 🔎 Unicode / Steganography Scan")
                u_col1, u_col2, u_col3 = st.columns(3)
                u_col1.metric("Invisible Characters", unicode_stats["removed_invisibles"])
                u_col2.metric("Exotic Spaces", unicode_stats["normalized_spaces"])
                u_col3.metric("Smart Quotes", unicode_stats["normalized_quotes"])
                st.warning("⚠️ Hidden invisible characters or exotic Unicode detected — potential watermarking or steganography.")
            else:
                st.success("✅ No invisible Unicode characters or steganographic markers detected.")




# =========================================================================== #
# TAB 5: Interactive User Guide & Help Studio
# =========================================================================== #

with tab_help:
    st.markdown("### 📖 Interactive User Guide & System Documentation")
    st.caption("Learn how to unlock the full potential of Writing Enhancer Pro. Drag the slider below to explore any section:")

    help_section = st.select_slider(
        "📍 Drag slider to navigate guides:",
        options=[
            "📄 Tab 1: Document Enhancer",
            "✏️ Tab 2: Quick Text Rewriter",
            "🛡️ Tab 3: Plagiarism & Citations",
            "🔬 Tab 4: AI Marker Auditor",
            "⚙️ Sidebar Controls & Parameters",
            "🛡️ Anti-AI & Turnitin Playbook",
            "❓ Frequently Asked Questions (FAQ)",
        ],
        value="📄 Tab 1: Document Enhancer",
        key="help_nav_slider",
    )

    st.markdown("---")

    # ------------------------------------------------------------- #
    # 1. Tab 1 Guide
    # ------------------------------------------------------------- #
    if help_section == "📄 Tab 1: Document Enhancer":
        st.markdown("### 📄 Tab 1: Document Enhancer & Instant Typesetter")
        st.info("💡 **Best For:** Complete research papers, thesis chapters, coursework assignments, and presentations (.docx, .pdf, .pptx, .txt, .md).")

        col_t1_a, col_t1_b = st.columns(2)
        with col_t1_a:
            st.markdown("""
#### 🌟 Two Ways to Process:
1. **📐 Format Document Only (0 API Cost — Instant 1-Second Execution):**
   - **No API key or tokens required.**
   - Instantly creates a publication-ready manuscript with:
     - Academic Title Page (with Course Code, Student ID, and Instructor name for Assignments).
     - Dynamic Table of Contents (TOC) with dot leaders (`...... Page X`).
     - Standard 1-inch margins (APA / Harvard / IEEE compliant).
     - Right-aligned running headers & centered page numbering.
     - Full paragraph justification (`JUSTIFY`) across Word and PDF.
     - Automatic Figure & Table caption detection.
2. **🚀 Humanize & Enhance Document (AI-Powered):**
   - Chunks large documents into safe word blocks (e.g. 400 words) with cross-chunk context stitching.
   - Eliminates AI detection markers (ZeroGPT, GPTZero, Turnitin).
   - Enhances vocabulary, academic register, and logical sentence flow.
""")
        with col_t1_b:
            st.markdown("""
#### 📋 Step-by-Step Instructions:
1. **Upload your document:** Drag and drop your `.docx`, `.pdf`, `.pptx`, `.txt`, or `.md` file.
2. **Review Step 1 Pre-Analysis Card:** The app scans your document and displays total words, headings, figure captions, table captions, and estimated page count.
3. **Choose your mode:** Click **"📐 Format Document Only"** for instant formatting or **"🚀 Humanize & Enhance Document"** for full AI rewriting.
4. **Inspect Results:** View Before/After comparison, 4-Gram originality scores, and the Inline Visual Word Diff (Red: removed AI words, Green: scholarly replacements).
5. **Download Formats:** Download in `.docx`, `.pdf`, `.md`, or `.txt`. Download buttons are preserved permanently and never disappear upon clicking!
""")

    # ------------------------------------------------------------- #
    # 2. Tab 2 Guide
    # ------------------------------------------------------------- #
    elif help_section == "✏️ Tab 2: Quick Text Rewriter":
        st.markdown("### ✏️ Tab 2: Quick Text & Multi-Document Rewriter")
        st.info("💡 **Best For:** Rewriting individual sections, paragraphs, abstracts, conclusions, or batch-merging multiple files.")

        c_t2_1, c_t2_2 = st.columns(2)
        with c_t2_1:
            st.markdown("""
#### 🌟 Core Features:
- **Direct Text Paste:** Paste any rough draft, messy AI text, or notes.
- **Multi-File Uploader:** Upload multiple `.docx`, `.pdf`, `.pptx`, `.txt`, and `.md` files at once.
  - **"📥 Load Selected File":** Loads an individual file into the editor.
  - **"📚 Merge & Load All Files":** Combines all uploaded files with section separators into one single manuscript for batch rewriting.
- **Section-Aware Processing:** Select whether your text is an Abstract, Introduction, Literature Review, Methodology, Results, Discussion, or Conclusion.
""")
        with c_t2_2:
            st.markdown("""
#### 🎯 The Scientific Conclusion Protocol:
Selecting the **Conclusion** section enforces the 4-part academic sequence:
1. **Restate the Main Idea:** Re-articulates the thesis using fresh vocabulary (never copied from introduction).
2. **Synthesize Key Findings:** Interconnects arguments rather than making a boring list.
3. **Answer the 'So What?':** Explains why the findings matter in the broader field.
4. **Forward-Looking Final Thought:** Leaves actionable future vectors or clinical/policy recommendations.
*Banned AI Clichés:* Words like *"In conclusion"*, *"To sum up"*, and *"All in all"* are automatically eradicated!
""")

    # ------------------------------------------------------------- #
    # 3. Tab 3 Guide
    # ------------------------------------------------------------- #
    elif help_section == "🛡️ Tab 3: Plagiarism & Citations":
        st.markdown("### 🛡️ Tab 3: Plagiarism Studio & Multi-Source Citation Authenticator")
        st.info("💡 **Best For:** Beating Turnitin / iThenticate string-matching, comparing drafts against multiple reference papers, and auto-generating citations.")

        st.markdown("""
#### 🛠️ Three Workflows Inside Tab 3:
""")
        wf1, wf2, wf3 = st.columns(3)
        with wf1:
            st.markdown("""
##### ⚡ 1. Deep Paraphraser
- Paste text or upload a file.
- Disrups consecutive 4-to-7 word N-grams that Turnitin flags.
- Displays **Originality Score %**, **4-Gram Overlap %**, and highlights identical matching sequences.
- Offers direct `.docx`, `.pdf`, `.md`, and `.txt` downloads.
""")
        with wf2:
            st.markdown("""
##### 🔍 2. Multi-Source Authenticator
- **Upload multiple reference PDFs / papers** (your research corpus).
- **Upload or paste your draft**.
- Compares your draft against each source individually and combined.
- Generates a **Source-by-Source Breakdown Table**.
- Auto-generates in-text citations & full bibliography (**APA**, **IEEE [1]**, or **Harvard**).
- 1-Click download of draft with references attached!
""")
        with wf3:
            st.markdown("""
##### 📚 3. Paraphrasing Masterclass
- Built-in educational guide teaching the 4 scientific pillars:
  1. *Read & Shield Technique*
  2. *Syntactic Flipping (Clause Inversion)*
  3. *Part-of-Speech Transformation (Nominalization)*
  4. *The Inviolable Citation Rule* (citations are mandatory even with 0% similarity).
""")

    # ------------------------------------------------------------- #
    # 4. Tab 4 Guide
    # ------------------------------------------------------------- #
    elif help_section == "🔬 Tab 4: AI Marker Auditor":
        st.markdown("### 🔬 Tab 4: AI Marker & Watermark Auditor")
        st.info("💡 **Best For:** Diagnostic text check before submitting to universities, journals, or Turnitin. 100% Free — No API Key needed!")

        c_t4_a, c_t4_b = st.columns(2)
        with c_t4_a:
            st.markdown("""
#### 🔍 Diagnostic Audits Performed:
- **AI Vulnerability Score (0 to 100):** Evaluates passive voice density, cliché academic transitions (*"furthermore"*, *"moreover"*, *"delve"*), and synthetic rhythm.
- **Em-Dash (`—`) & Hyphen Frequency:** ZeroGPT and Turnitin penalize em-dash parentheticals (`—word—`). This audit counts all em-dashes, en-dashes, and double hyphens.
- **Sentence Burstiness Meter (Rhythm Variance):** Calculates the standard deviation of sentence lengths. Natural human writing mixes 5-word sentences with 30-word sentences (Std Dev $> 7.0$). AI writing is robotic and flat (Std Dev $< 4.0$).
- **Unicode & Steganography Scanner:** Detects zero-width invisible characters (`\u200b`, `\ufeff`) often used by LLM providers as synthetic watermarks.
""")
        with c_t4_b:
            st.markdown("""
#### ✨ 1-Click Anti-AI Sanitizer:
- Don't want to spend tokens rewriting?
- Simply paste your text and click **"✨ 1-Click Anti-AI Sanitizer"**.
- The app automatically:
  - Converts all em-dashes (`—`) and double hyphens (`--`) into clean human-style commas or split sentences.
  - Scrubs all invisible zero-width watermarks and exotic Unicode characters.
  - Allows instant download of the sanitized text in `.docx` or `.txt`!
""")

    # ------------------------------------------------------------- #
    # 5. Sidebar Guide
    # ------------------------------------------------------------- #
    elif help_section == "⚙️ Sidebar Controls & Parameters":
        st.markdown("### ⚙️ Sidebar Parameters & Academic Controls Guide")
        st.info("💡 **Best For:** Customizing output style, document tone, formatting rules, and API keys.")

        with st.expander("🔑 1. API Provider & Model Selection", expanded=True):
            st.markdown("""
- **Google Gemini (Recommended — 100% Free):** Get a free key at [Google AI Studio](https://aistudio.google.com/apikey). No credit card required. Free tier offers generous requests per minute.
  - *Models:* `gemini-2.5-flash` (recommended default), `gemini-2.5-flash-lite`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`.
- **OpenRouter (Multi-Model):** Single key for Claude 3.5 Sonnet, GPT-4o, and Llama 3.3.
- **OpenAI:** Direct OpenAI API keys for GPT-4o and GPT-4o-mini.
""")

        with st.expander("📋 2. Processing Checkboxes (What Each Option Does)", expanded=True):
            st.markdown("""
| Option | What It Does When Checked |
| :--- | :--- |
| **Humanize (Remove AI patterns)** | Strips synthetic transitional clichés (*"it is crucial to note"*, *"delve into"*), robotic preambles, and formulaic endings. |
| **Academic Tone & Lexicon** | Elevates colloquial phrasing into scholarly vocabulary and disciplined academic register. |
| **Anti-Plagiarism (N-Gram Breaking)** | Rearranges clauses and transposes words to defeat 4-to-7 word string matching in Turnitin / iThenticate. |
| **Flow & Paragraph Stitching** | Uses cross-chunk context stitching so paragraphs connect smoothly without repetitive opening words. |
| **Sentence Structure (Burstiness)** | Interleaves short punchy sentences (5–8 words) with complex academic sentences (24–34 words) to beat GPTZero. |
| **Tables & Figures Integrity** | Protects markdown tables, LaTeX equations, numbers, and captions so factual data is never altered. |
""")

        with st.expander("🎯 3. Academic Work Mode & Document Type", expanded=False):
            st.markdown("""
- **📄 Journal Article (IMRaD Standard):** High-impact peer-reviewed paper format following Introduction, Methods, Results, and Discussion.
- **🎓 Thesis / PhD Dissertation:** Exhaustive scholarly depth with extensive theoretical framework and methodological justification.
- **📚 Coursework / Student Assignment:** Formatted for student submissions with fields for **Course Code**, **Student ID / Roll Number**, and **Instructor Name** on the Title Page.
- **📖 Systematic Literature Review:** Focuses on thematic categorization, meta-analysis, and contrasting scholarly viewpoints.
- **💻 Conference Paper (IEEE / ACM):** High technical density, direct contributions, and concise empirical arguments.
- **💡 Grant Proposal / Scientific Pitch:** Persuasive yet rigorous problem significance, impact metrics, and methodology feasibility.
- **📝 Academic Essay / Term Paper:** Clear thesis defense and pedagogical argumentative flow.
""")

        with st.expander("🌐 4. English Tone & Regional Conventions", expanded=False):
            st.markdown("""
- **Academic Rigorous (Global High-Impact):** Disciplined, objective third-person scholarly prose.
- **Professional & Executive:** Direct, clear, active-voice communication suitable for reports.
- **Native US English (APA Standard):** American spelling (`-ize`, `-or`), APA 7th edition punctuation.
- **Native UK / Oxford English:** British spelling (`-ise`, `-our`), Oxford comma convention.
- **Indo-Pak / Global Scholarly English (Normalizer):** Specially engineered for Pakistani and South Asian scholars. Automatically translates regional idioms into internationally recognized academic English:
  - *"prepone"* $\rightarrow$ *"rescheduled to an earlier date"*
  - *"do the needful"* $\rightarrow$ *"take the requisite measures"*
  - *"revert back"* $\rightarrow$ *"provide a formal response / reply"*
  - *"passed out from university"* $\rightarrow$ *"graduated from university"*
  - *"intimate the committee"* $\rightarrow$ *"notify the committee"*
""")

        with st.expander("📐 5. Formatting Checklist & Alignment", expanded=False):
            st.markdown("""
- **Paragraph Alignment:** Choose **Justified (Default Academic)**, Left, Center, or Right. Standard academic publications (APA, IEEE, Harvard) require full justification.
- **Title Page:** Generates a formal cover page followed by a clean Page Break.
- **Table of Contents (TOC):** Automatically builds leader-dot page rows (`...... Page X`) and native Word XML dynamic fields.
- **1-Inch Standard Margins:** 72pt margins on top, bottom, left, and right.
- **Typography Preset:** Times New Roman (APA Standard), Arial, Calibri, Georgia, or Garamond.
""")

    # ------------------------------------------------------------- #
    # 6. Anti-AI Playbook
    # ------------------------------------------------------------- #
    elif help_section == "🛡️ Anti-AI & Turnitin Playbook":
        st.markdown("### 🛡️ Anti-AI & Turnitin Evasion Playbook")
        st.info("💡 **The Science of Beating Detectors:** Learn how Turnitin, ZeroGPT, GPTZero, Copyleaks, and Winston AI work — and how to pass them with 0% AI and 0% similarity.")

        st.markdown("""
#### 🏆 The 5 Golden Rules of Human Academic Writing:

1. **Rule #1: Absolute Ban on Em-Dashes (`—`) and Double Hyphens (`--`)**
   - AI models use em-dashes for non-restrictive clauses far more frequently than human writers.
   - ZeroGPT and Turnitin assign heavy penalty points to em-dashes. Use commas, parentheses, or separate sentences instead.

2. **Rule #2: Enforce Radical Burstiness (Sentence Length Variety)**
   - AI writes sentences of nearly identical lengths (e.g. 17 words, 18 words, 17 words).
   - Humans naturally alternate: write one short 6-word sentence. Then follow it with a 30-word compound sentence containing semicolons and dependent clauses. Target a standard deviation $> 8.0$.

3. **Rule #3: Maximize Perplexity (Avoid Greedily Predictable Words)**
   - Detectors calculate how 'predictable' each next word is.
   - Banned predictable bridges: *"In today's fast-paced world"*, *"It is important to remember"*, *"A testament to"*, *"Tapestry of"*, *"Delve into"*.
   - Replace them with precise, field-specific empirical terminology.

4. **Rule #4: Break 4-to-6 Word N-Grams with Syntactic Flipping**
   - Turnitin compares consecutive runs of 4 to 6 words against 99+ billion archived pages.
   - Flip cause-and-effect clauses:
     - *Original:* "Because the sample degraded, the experiment failed."
     - *Flipped:* "Experimental failure directly resulted from rapid sample degradation."

5. **Rule #5: Scrub Invisible Zero-Width Steganography**
   - Some AI engines inject invisible zero-width spaces (`\u200b`, `\ufeff`) as digital watermarks.
   - Always run your text through our built-in Unicode cleaner or Tab 4 sanitizer before submission.
""")

    # ------------------------------------------------------------- #
    # 7. FAQ
    # ------------------------------------------------------------- #
    elif help_section == "❓ Frequently Asked Questions (FAQ)":
        st.markdown("### ❓ Frequently Asked Questions (FAQ)")

        with st.expander("Q1: Is Google Gemini API really 100% free?", expanded=True):
            st.markdown("""
**Yes!** Google AI Studio provides a free tier for developers and researchers that does not require any credit card. You get generous limits (15 requests per minute), which is more than enough to enhance and humanize entire manuscripts.
- Get your key here: [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
""")

        with st.expander("Q2: Can I use this app without an API Key?", expanded=True):
            st.markdown("""
**Yes, absolutely!** You can use two major features without any API key:
1. **📐 Format Document Only (in Tab 1):** Instantly creates professional Title Pages, Table of Contents, 1-inch margins, and full justification in Word & PDF with 0 tokens and 0 API cost.
2. **🔬 AI Marker Auditor (in Tab 4):** Diagnoses AI score, burstiness, em-dashes, and scrubs watermarks 100% offline with no API key.
""")

        with st.expander("Q3: What should I do if Turnitin flags similarity?", expanded=False):
            st.markdown("""
1. Go to **Tab 3 ("Plagiarism & Paraphrasing Studio")**.
2. Select **"Similarity & Overlap Inspector"**.
3. Upload your reference papers in the Left Box and paste your draft in the Right Box.
4. The system will pinpoint the exact 4-word sequences that caused the flag and provide suggested academic citations (**APA**, **IEEE**, or **Harvard**) to ensure complete academic integrity.
""")


# =========================================================================== #
# Footer
# =========================================================================== #

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888; font-size: 0.85em;'>"
    "Writing Enhancer Pro &nbsp;•&nbsp; Multi-Provider LLM Engine (Gemini / OpenAI / OpenRouter) &nbsp;•&nbsp; "
    "Anti-AI Humanizer &nbsp;•&nbsp; N-Gram Plagiarism Disruption &nbsp;•&nbsp; "
    "Academic Tone & Flow Enforcement"
    "</div>",
    unsafe_allow_html=True,
)

