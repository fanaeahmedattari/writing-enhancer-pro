"""
core/docx_parser.py
Production-ready Document Parser & Smart Chunking Engine for .docx (and multi-format documents).
Features:
- Document-order element traversal (Paragraphs, Headings, Bullet Lists, and Tables)
- Automatic academic section classification (Introduction, Literature Review, Methodology, Results, Discussion, Conclusion)
- Intelligent word-limited chunking with sentence-boundary integrity
- Hook & Eye context stitching across chunk boundaries
- Figure and Table scanner with caption & in-text citation analysis
- Container metadata scrubbing (strips author, revision, and generator tags)
- Clean, structured document reassembly preserving headings, tables, and lists
"""

from __future__ import annotations

import os
import re
from datetime import datetime
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import docx
from docx.document import Document as DocxDocument
from docx.oxml.table import CT_Tbl
from docx.oxml.text.paragraph import CT_P
from docx.table import Table, _Cell
from docx.text.paragraph import Paragraph
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import qn, nsdecls

# Academic section keyword mapping
ACADEMIC_SECTION_MAP: Dict[str, List[str]] = {
    "abstract": ["abstract", "executive summary"],
    "introduction": ["introduction", "background", "motivation", "research problem"],
    "literature_review": ["literature review", "related work", "previous studies", "theoretical framework", "state of the art"],
    "methodology": ["methodology", "methods", "materials and methods", "experimental setup", "proposed framework", "model architecture", "data collection"],
    "results": ["results", "findings", "experimental results", "empirical analysis", "evaluation"],
    "discussion": ["discussion", "implications", "comparative analysis", "limitations"],
    "conclusion": ["conclusion", "conclusions", "concluding remarks", "summary", "future work"],
}

# Regex patterns for figures, tables, and academic citations
FIGURE_CAPTION_RE = re.compile(r"^\s*(?:figure|fig\.?)\s*(\d+[a-zA-Z]?(?:[-.]\d+)?)?\s*[:.-]?\s+(.*)$", re.IGNORECASE)
TABLE_CAPTION_RE = re.compile(r"^\s*(?:table|tbl\.?)\s*(\d+[a-zA-Z]?(?:[-.]\d+)?)?\s*[:.-]?\s+(.*)$", re.IGNORECASE)
IN_TEXT_FIGURE_RE = re.compile(r"\b(?:figure|fig\.?)\s*(\d+[a-zA-Z]?(?:[-.]\d+)?)\b", re.IGNORECASE)
IN_TEXT_TABLE_RE = re.compile(r"\b(?:table|tbl\.?)\s*(\d+[a-zA-Z]?(?:[-.]\d+)?)\b", re.IGNORECASE)


def iter_block_items(parent: Union[DocxDocument, _Cell]) -> Generator[Union[Paragraph, Table], None, None]:
    """
    Yield each paragraph and table child within *parent*, in strict document order.
    """
    if isinstance(parent, DocxDocument):
        parent_elm = parent.element.body
    elif isinstance(parent, _Cell):
        parent_elm = parent._tc
    else:
        raise ValueError("Unsupported parent type for block traversal.")

    for child in parent_elm.iterchildren():
        if isinstance(child, CT_P):
            yield Paragraph(child, parent)
        elif isinstance(child, CT_Tbl):
            yield Table(child, parent)


def detect_section_type(heading_text: str) -> str:
    """
    Classifies a heading string into a standardized academic section type.
    """
    clean = heading_text.lower().strip()
    # Strip leading numbering like '1.', '2.1', 'III.', 'Section 1:', 'Chapter 2:'
    clean = re.sub(r"^(?:(?:section|chapter|chap\.?)\s+)?(?:(?:\d+(?:\.\d+)*)|(?:[ivxlcdm]+))[\.\s\-–—:]+\s*", "", clean, flags=re.IGNORECASE).strip()

    for sec_type, keywords in ACADEMIC_SECTION_MAP.items():
        for kw in keywords:
            if re.search(r"\b" + re.escape(kw) + r"\b", clean):
                return sec_type
    return "general"




def format_table_to_markdown(table: Table) -> str:
    """
    Converts a docx Table object into a clean Markdown table string.
    """
    rows_data = []
    for row in table.rows:
        row_cells = [cell.text.replace("\n", " ").strip() for cell in row.cells]
        rows_data.append(row_cells)

    if not rows_data:
        return ""

    num_cols = max(len(r) for r in rows_data)
    # Pad shorter rows if any
    for r in rows_data:
        while len(r) < num_cols:
            r.append("")

    header_row = rows_data[0]
    separator_row = ["---"] * num_cols

    md_lines = [
        "| " + " | ".join(header_row) + " |",
        "| " + " | ".join(separator_row) + " |"
    ]

    for data_row in rows_data[1:]:
        md_lines.append("| " + " | ".join(data_row) + " |")

    return "\n".join(md_lines)


class DocxChunker:
    """
    Parses .docx files into manageable chunks while preserving structural 
    metadata (headings, list items, normal text, tables) and reassembles them post-processing.
    """

    def __init__(self, max_words_per_chunk: int = 400):
        self.max_words_per_chunk = max_words_per_chunk

    def parse_docx(self, docx_path: str) -> List[Dict[str, Any]]:
        """
        Extracts document elements into structured chunks with formatting metadata,
        section detection, and Hook-and-Eye context stitching.
        """
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"Document file not found: {docx_path}")

        doc = docx.Document(docx_path)
        chunks: List[Dict[str, Any]] = []
        current_elements: List[Dict[str, Any]] = []
        current_word_count = 0
        chunk_index = 0
        current_section = "general"
        prev_tail_sentence = ""
        para_idx = 0

        for block in iter_block_items(doc):
            if isinstance(block, Paragraph):
                raw_text = block.text.strip()
                has_drawing = bool(block._element.xpath('.//w:drawing') or block._element.xpath('.//w:pict'))
                has_math = bool(block._element.xpath('.//m:oMath') or block._element.xpath('.//m:oMathPara'))

                if not raw_text and not has_drawing and not has_math:
                    para_idx += 1
                    continue

                style_name = block.style.name if block.style else "Normal"
                is_heading = style_name.lower().startswith("heading") or style_name.lower() == "title"
                
                # Determine heading level
                heading_level = 1
                if is_heading:
                    level_match = re.search(r"heading\s*(\d+)", style_name, re.IGNORECASE)
                    if level_match:
                        heading_level = int(level_match.group(1))
                    # Update active section if major heading
                    if heading_level <= 2:
                        detected_sec = detect_section_type(raw_text)
                        if detected_sec != "general":
                            current_section = detected_sec

                is_list = style_name.lower().startswith("list") or raw_text.startswith(("- ", "* ", "• "))
                is_caption = bool(FIGURE_CAPTION_RE.match(raw_text) or TABLE_CAPTION_RE.match(raw_text))

                word_count = len(raw_text.split())

                element = {
                    "type": "heading" if is_heading else ("list_item" if is_list else "paragraph"),
                    "text": raw_text,
                    "style": style_name,
                    "level": heading_level if is_heading else 0,
                    "is_caption": is_caption,
                    "has_drawing": has_drawing,
                    "has_math": has_math,
                    "para_idx": para_idx,
                    "section": current_section,
                    "word_count": word_count
                }
                para_idx += 1

                # Boundary rule: If a major Heading appears and we have reached at least 200 words,
                # or if adding this element exceeds max_words_per_chunk, push the current chunk.
                should_split = (
                    current_elements and (
                        (current_word_count + word_count > self.max_words_per_chunk) or
                        (is_heading and heading_level <= 2 and current_word_count >= 200)
                    )
                )

                if should_split:
                    chunk_text = self._assemble_elements_text(current_elements)
                    tail_sent = self._extract_tail_sentence(chunk_text)
                    chunk_sec = current_elements[0].get("section", "general") if current_elements else current_section
                    
                    chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "section_type": chunk_sec,
                        "elements": current_elements,
                        "para_indices": [el["para_idx"] for el in current_elements if "para_idx" in el],
                        "prev_context_tail": prev_tail_sentence,
                        "has_table": any(el["type"] == "table" for el in current_elements),
                        "has_figures": any(el.get("is_caption") or el.get("has_drawing") for el in current_elements)
                    })
                    
                    prev_tail_sentence = tail_sent
                    chunk_index += 1
                    current_elements = [element]
                    current_word_count = word_count
                else:
                    current_elements.append(element)
                    current_word_count += word_count

            elif isinstance(block, Table):
                # Process table block
                md_table = format_table_to_markdown(block)
                if not md_table.strip():
                    continue

                table_word_count = len(md_table.split())
                element = {
                    "type": "table",
                    "text": md_table,
                    "style": "Table",
                    "level": 0,
                    "is_caption": False,
                    "section": current_section,
                    "word_count": table_word_count,
                    "raw_table": block
                }

                if current_elements and (current_word_count + table_word_count > self.max_words_per_chunk):
                    chunk_text = self._assemble_elements_text(current_elements)
                    tail_sent = self._extract_tail_sentence(chunk_text)
                    chunk_sec = current_elements[0].get("section", "general") if current_elements else current_section
                    
                    chunks.append({
                        "chunk_id": chunk_index,
                        "text": chunk_text,
                        "word_count": current_word_count,
                        "section_type": chunk_sec,
                        "elements": current_elements,
                        "para_indices": [el["para_idx"] for el in current_elements if "para_idx" in el],
                        "prev_context_tail": prev_tail_sentence,
                        "has_table": any(el["type"] == "table" for el in current_elements),
                        "has_figures": any(el.get("is_caption") or el.get("has_drawing") for el in current_elements)
                    })
                    
                    prev_tail_sentence = tail_sent
                    chunk_index += 1
                    current_elements = [element]
                    current_word_count = table_word_count
                else:
                    current_elements.append(element)
                    current_word_count += table_word_count

        # Append remaining elements
        if current_elements:
            chunk_text = self._assemble_elements_text(current_elements)
            chunk_sec = current_elements[0].get("section", "general") if current_elements else current_section
            chunks.append({
                "chunk_id": chunk_index,
                "text": chunk_text,
                "word_count": current_word_count,
                "section_type": chunk_sec,
                "elements": current_elements,
                "para_indices": [el["para_idx"] for el in current_elements if "para_idx" in el],
                "prev_context_tail": prev_tail_sentence,
                "has_table": any(el["type"] == "table" for el in current_elements),
                "has_figures": any(el.get("is_caption") or el.get("has_drawing") for el in current_elements)
            })


        return chunks

    def extract_global_document_context(self, docx_path: str) -> str:
        """
        Extracts high-level macro-context (Title, primary headings, core targets,
        and defined acronyms) from the document to serve as the 'North Star'
        passed to every chunk during rewriting.
        """
        if not os.path.exists(docx_path):
            return ""

        doc = docx.Document(docx_path)
        title = ""
        headings: List[str] = []
        acronyms_found: Set[str] = set()

        acronym_re = re.compile(r'\(([A-Z]{2,6})\)')

        for p in doc.paragraphs[:40]:
            txt = p.text.strip()
            if not txt:
                continue
            style_name = p.style.name.lower() if p.style else ""
            if not title and (style_name == "title" or (len(txt.split()) in range(4, 25) and not txt.endswith("."))):
                title = txt
            elif style_name.startswith("heading"):
                headings.append(txt)
            
            for ac in acronym_re.findall(txt):
                acronyms_found.add(ac)

        context_lines = []
        if title:
            context_lines.append(f"Manuscript Title: {title}")
        if headings:
            context_lines.append(f"Major Sections / Headings: {', '.join(headings[:5])}")
        if acronyms_found:
            context_lines.append(f"Established Acronyms: {', '.join(sorted(list(acronyms_found))[:10])}")

        return "\n".join(context_lines)

    def scan_figures_and_tables(self, docx_path: str) -> Dict[str, Any]:
        """
        Scans document for tables, figure captions, and body text references to ensure
        academic citation completeness.
        """
        if not os.path.exists(docx_path):
            raise FileNotFoundError(f"File not found: {docx_path}")

        doc = docx.Document(docx_path)
        tables_info: List[Dict[str, Any]] = []
        figure_captions: List[Dict[str, Any]] = []
        table_captions: List[Dict[str, Any]] = []
        in_text_fig_refs: set[str] = set()
        in_text_tbl_refs: set[str] = set()

        # Count tables
        for idx, tbl in enumerate(doc.tables, start=1):
            num_rows = len(tbl.rows)
            num_cols = len(tbl.columns) if tbl.rows else 0
            tables_info.append({
                "table_index": idx,
                "rows": num_rows,
                "columns": num_cols
            })

        # Scan paragraphs
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue

            fig_match = FIGURE_CAPTION_RE.match(text)
            if fig_match:
                fig_id = (fig_match.group(1) or "").strip()
                caption_text = (fig_match.group(2) or "").strip()
                figure_captions.append({
                    "id": fig_id,
                    "caption": caption_text,
                    "full_text": text
                })
                continue

            tbl_match = TABLE_CAPTION_RE.match(text)
            if tbl_match:
                tbl_id = (tbl_match.group(1) or "").strip()
                caption_text = (tbl_match.group(2) or "").strip()
                table_captions.append({
                    "id": tbl_id,
                    "caption": caption_text,
                    "full_text": text
                })
                continue

            # Body text citation scan
            for f_match in IN_TEXT_FIGURE_RE.finditer(text):
                matched_fig = f_match.group(1)
                if matched_fig:
                    in_text_fig_refs.add(matched_fig.lower().strip())
            for t_match in IN_TEXT_TABLE_RE.finditer(text):
                matched_tbl = t_match.group(1)
                if matched_tbl:
                    in_text_tbl_refs.add(matched_tbl.lower().strip())

        caption_fig_ids = {fc["id"].lower() for fc in figure_captions if fc.get("id")}
        caption_tbl_ids = {tc["id"].lower() for tc in table_captions if tc.get("id")}

        missing_fig_citations = list(caption_fig_ids - in_text_fig_refs)
        missing_tbl_citations = list(caption_tbl_ids - in_text_tbl_refs)
        uncaptioned_fig_refs = list(in_text_fig_refs - caption_fig_ids)
        uncaptioned_tbl_refs = list(in_text_tbl_refs - caption_tbl_ids)

        advice: List[str] = []
        if missing_fig_citations:
            advice.append(f"Figure(s) {', '.join(missing_fig_citations)} have captions but are not referenced in the body text.")
        if missing_tbl_citations:
            advice.append(f"Table(s) {', '.join(missing_tbl_citations)} have captions but are not referenced in the body text.")
        if uncaptioned_fig_refs:
            advice.append(f"Figure(s) {', '.join(uncaptioned_fig_refs)} are cited in text but lack corresponding caption lines.")
        if uncaptioned_tbl_refs:
            advice.append(f"Table(s) {', '.join(uncaptioned_tbl_refs)} are cited in text but lack corresponding caption lines.")

        return {
            "total_tables": len(tables_info),
            "tables_structure": tables_info,
            "total_figure_captions": len(figure_captions),
            "figure_captions": figure_captions,
            "total_table_captions": len(table_captions),
            "table_captions": table_captions,
            "in_text_figure_citations": sorted(list(in_text_fig_refs)),
            "in_text_table_citations": sorted(list(in_text_tbl_refs)),
            "missing_figure_citations": missing_fig_citations,
            "missing_table_citations": missing_tbl_citations,
            "uncaptioned_figure_references": uncaptioned_fig_refs,
            "uncaptioned_table_references": uncaptioned_tbl_refs,
            "academic_advice": advice
        }

    def reassemble_docx_preserving_media(
        self,
        original_docx_path: str,
        humanized_chunks: Union[List[str], List[Dict[str, Any]]],
        output_path: str,
        typography_preset: str = "Times New Roman",
        margins_inches: float = 1.0,
        line_spacing: float = 1.5,
        alignment: str = "JUSTIFY",
        include_title_page: bool = False,
        title_page_data: Optional[Dict[str, str]] = None,
        include_toc: bool = False,
    ) -> str:
        """
        Reconstructs the enhanced document IN-PLACE inside the original .docx package,
        preserving 100% of all embedded drawings, figures, images, math equations, tables,
        and header/footer structures.
        """
        if not os.path.exists(original_docx_path):
            raise FileNotFoundError(f"Original file not found for media preservation: {original_docx_path}")

        doc = docx.Document(original_docx_path)
        para_map = {i: p for i, p in enumerate(doc.paragraphs)}

        # 1. Apply Academic Margins
        for section in doc.sections:
            section.top_margin = Inches(margins_inches)
            section.bottom_margin = Inches(margins_inches)
            section.left_margin = Inches(margins_inches)
            section.right_margin = Inches(margins_inches)

        align_map = {
            "JUSTIFY": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "JUSTIFIED": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "LEFT": WD_ALIGN_PARAGRAPH.LEFT,
            "CENTER": WD_ALIGN_PARAGRAPH.CENTER,
            "RIGHT": WD_ALIGN_PARAGRAPH.RIGHT,
        }
        chosen_align = align_map.get(alignment.strip().upper().split()[0], WD_ALIGN_PARAGRAPH.JUSTIFY)

        # 2. Configure Typography Styles
        if "Normal" in doc.styles:
            normal = doc.styles["Normal"]
            normal.font.name = typography_preset
            normal.font.size = Pt(12)
            normal.paragraph_format.line_spacing = line_spacing

        # 3. Update paragraphs in-place for each chunk
        for ch in humanized_chunks:
            if not isinstance(ch, dict):
                continue

            humanized_text = ch.get("humanized_text") or ch.get("text") or ""
            if not humanized_text.strip():
                continue

            elements = ch.get("elements", [])
            # Find eligible paragraphs: no drawings, no math, no table
            target_indices: List[int] = []
            for el in elements:
                p_idx = el.get("para_idx")
                if p_idx is not None and p_idx in para_map:
                    p = para_map[p_idx]
                    has_drawing = bool(p._element.xpath('.//w:drawing') or p._element.xpath('.//w:pict'))
                    has_math = bool(p._element.xpath('.//m:oMath') or p._element.xpath('.//m:oMathPara'))
                    if not has_drawing and not has_math:
                        target_indices.append(p_idx)

            if not target_indices:
                continue

            # Split rewritten text into paragraphs
            rewritten_paras = [p.strip() for p in humanized_text.split("\n\n") if p.strip()]

            # Update target paragraphs
            for i, p_idx in enumerate(target_indices):
                p = para_map.get(p_idx)
                if p is None:
                    continue
                if i < len(rewritten_paras):
                    para_text = rewritten_paras[i]
                    is_h = para_text.startswith("#")
                    if is_h:
                        clean_heading = re.sub(r"^#{1,6}\s*", "", para_text).strip()
                        p.text = clean_heading
                        p.alignment = WD_ALIGN_PARAGRAPH.LEFT
                    else:
                        p.text = para_text
                        p.alignment = chosen_align
                    for run in p.runs:
                        run.font.name = typography_preset
                else:
                    # Blank out extra paragraphs in this chunk without breaking XML structure
                    p.text = ""

            # If there are MORE rewritten paragraphs than original target paragraphs,
            # dynamically insert the extra paragraphs right after the last target paragraph
            if len(rewritten_paras) > len(target_indices) and target_indices:
                last_p = para_map.get(target_indices[-1])
                if last_p is not None:
                    curr_p = last_p
                    for extra_text in rewritten_paras[len(target_indices):]:
                        new_p_elem = OxmlElement('w:p')
                        curr_p._p.addnext(new_p_elem)
                        new_p_obj = docx.text.paragraph.Paragraph(new_p_elem, doc)

                        is_h = extra_text.startswith("#")
                        if is_h:
                            clean_h = re.sub(r"^#{1,6}\s*", "", extra_text).strip()
                            new_p_obj.text = clean_h
                            new_p_obj.alignment = WD_ALIGN_PARAGRAPH.LEFT
                        else:
                            new_p_obj.text = extra_text
                            new_p_obj.alignment = chosen_align
                        for run in new_p_obj.runs:
                            run.font.name = typography_preset
                        curr_p = new_p_obj

        self.strip_document_metadata(doc)
        doc.save(output_path)
        return output_path

    def reassemble_docx(
        self,
        original_docx_path: str,
        humanized_chunks: Union[List[str], List[Dict[str, Any]]],
        output_path: str,
        preserve_original_media: bool = True,
        include_title_page: bool = False,
        title_page_data: Optional[Dict[str, str]] = None,
        include_toc: bool = True,
        typography_preset: str = "Times New Roman",
        margins_inches: float = 1.0,
        line_spacing: float = 1.5,
        running_head: str = "Writing Enhancer Pro — Academic Manuscript",
        alignment: str = "JUSTIFY",
    ) -> str:
        """
        Reconstructs a publication-ready .docx document.
        If preserve_original_media=True, retains 100% of embedded images, graphics, and formulas in-place.
        """
        if preserve_original_media and os.path.exists(original_docx_path) and any(isinstance(c, dict) and "para_indices" in c for c in humanized_chunks):
            return self.reassemble_docx_preserving_media(
                original_docx_path=original_docx_path,
                humanized_chunks=humanized_chunks,
                output_path=output_path,
                typography_preset=typography_preset,
                margins_inches=margins_inches,
                line_spacing=line_spacing,
                alignment=alignment,
                include_title_page=include_title_page,
                title_page_data=title_page_data,
                include_toc=include_toc,
            )

        new_doc = docx.Document()

        # Alignment mapping
        align_map = {
            "JUSTIFY": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "JUSTIFIED": WD_ALIGN_PARAGRAPH.JUSTIFY,
            "LEFT": WD_ALIGN_PARAGRAPH.LEFT,
            "CENTER": WD_ALIGN_PARAGRAPH.CENTER,
            "RIGHT": WD_ALIGN_PARAGRAPH.RIGHT,
        }
        clean_align_key = alignment.strip().upper().split()[0]
        chosen_align = align_map.get(clean_align_key, WD_ALIGN_PARAGRAPH.JUSTIFY)

        # --- 1. Apply Academic Margins & Running Header/Footer ---
        for section in new_doc.sections:
            section.top_margin = Inches(margins_inches)
            section.bottom_margin = Inches(margins_inches)
            section.left_margin = Inches(margins_inches)
            section.right_margin = Inches(margins_inches)

            # Running Header (Right-Aligned)
            header = section.header
            if header.paragraphs:
                hp = header.paragraphs[0]
                hp.alignment = WD_ALIGN_PARAGRAPH.RIGHT
                hrun = hp.add_run(running_head)
                hrun.font.name = typography_preset
                hrun.font.size = Pt(8.5)
                hrun.font.italic = True
                hrun.font.color.rgb = RGBColor(128, 128, 128)

            # Running Footer (Centered Page Number)
            footer = section.footer
            if footer.paragraphs:
                fp = footer.paragraphs[0]
                fp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                frun = fp.add_run("Page ")
                frun.font.name = typography_preset
                frun.font.size = Pt(9.0)
                frun.font.color.rgb = RGBColor(128, 128, 128)
                fldChar1 = OxmlElement('w:fldChar')
                fldChar1.set(qn('w:fldCharType'), 'begin')
                instrText = OxmlElement('w:instrText')
                instrText.set(qn('xml:space'), 'preserve')
                instrText.text = 'PAGE'
                fldChar2 = OxmlElement('w:fldChar')
                fldChar2.set(qn('w:fldCharType'), 'separate')
                fldChar3 = OxmlElement('w:fldChar')
                fldChar3.set(qn('w:fldCharType'), 'end')
                frun._r.append(fldChar1)
                frun._r.append(instrText)
                frun._r.append(fldChar2)
                frun._r.append(fldChar3)

        # --- 2. Configure Typography Styles ---
        styles = new_doc.styles
        if "Normal" in styles:
            normal = styles["Normal"]
            normal.font.name = typography_preset
            normal.font.size = Pt(12)
            normal.font.color.rgb = RGBColor(33, 33, 33)
            normal.paragraph_format.line_spacing = line_spacing
            normal.paragraph_format.space_after = Pt(6)
            normal.paragraph_format.alignment = chosen_align

        # --- 2.5. Generate Formal Academic Title Page ---
        if include_title_page:
            data = title_page_data or {}
            doc_title = data.get("title", "").strip() or "Academic Research Manuscript"
            paper_type = data.get("paper_type", "").strip() or "Research Paper"
            author_name = data.get("author", "").strip() or "Author Name"
            student_id = data.get("student_id", "").strip()
            course = data.get("course", "").strip()
            instructor = data.get("instructor", "").strip()
            affiliation = data.get("institution", "").strip() or "Department / Academic Institution"
            doc_date = data.get("date", "").strip() or datetime.now().strftime("%B %Y")

            top_p = new_doc.add_paragraph()
            top_p.paragraph_format.space_before = Pt(60 if (student_id or course or instructor) else 80)

            tp = new_doc.add_paragraph()
            tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
            tp.paragraph_format.space_after = Pt(12)
            trun = tp.add_run(doc_title)
            trun.font.name = typography_preset
            trun.font.size = Pt(22)
            trun.font.bold = True
            trun.font.color.rgb = RGBColor(26, 35, 126)

            sub_p = new_doc.add_paragraph()
            sub_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            sub_p.paragraph_format.space_after = Pt(24 if course else 36)
            srun = sub_p.add_run(paper_type)
            srun.font.name = typography_preset
            srun.font.size = Pt(13)
            srun.font.italic = True
            srun.font.color.rgb = RGBColor(100, 100, 100)

            # Course details for Student Assignments
            if course:
                cp = new_doc.add_paragraph()
                cp.alignment = WD_ALIGN_PARAGRAPH.CENTER
                cp.paragraph_format.space_after = Pt(16)
                crun = cp.add_run(f"Course: {course}")
                crun.font.name = typography_preset
                crun.font.size = Pt(12)
                crun.font.bold = True
                crun.font.color.rgb = RGBColor(40, 53, 147)

            auth_p = new_doc.add_paragraph()
            auth_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            auth_p.paragraph_format.space_after = Pt(6)
            author_display = f"{author_name} (Student ID / Roll No: {student_id})" if student_id else author_name
            arun = auth_p.add_run(author_display)
            arun.font.name = typography_preset
            arun.font.size = Pt(13.5)
            arun.font.bold = True
            arun.font.color.rgb = RGBColor(33, 33, 33)

            # Instructor details for Student Assignments
            if instructor:
                ip = new_doc.add_paragraph()
                ip.alignment = WD_ALIGN_PARAGRAPH.CENTER
                ip.paragraph_format.space_after = Pt(6)
                irun = ip.add_run(f"Submitted to: {instructor}")
                irun.font.name = typography_preset
                irun.font.size = Pt(11.5)
                irun.font.color.rgb = RGBColor(66, 66, 66)

            aff_p = new_doc.add_paragraph()
            aff_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            aff_p.paragraph_format.space_after = Pt(6)
            aff_run = aff_p.add_run(affiliation)
            aff_run.font.name = typography_preset
            aff_run.font.size = Pt(11.5)
            aff_run.font.color.rgb = RGBColor(66, 66, 66)

            dt_p = new_doc.add_paragraph()
            dt_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            dt_p.paragraph_format.space_after = Pt(40)
            drun = dt_p.add_run(doc_date)
            drun.font.name = typography_preset
            drun.font.size = Pt(11)
            drun.font.color.rgb = RGBColor(128, 128, 128)

            new_doc.add_page_break()


        # --- 3. Extract Headings & Content Blocks ---
        processed_texts: List[str] = []
        for ch in humanized_chunks:
            if isinstance(ch, dict):
                val = ch.get("humanized_text") or ch.get("text") or ""
                processed_texts.append(val)
            elif isinstance(ch, str):
                processed_texts.append(ch)

        full_processed_text = "\n\n".join(processed_texts)
        norm_text = re.sub(r"(?m)^(#{1,6}\s+.*)$", r"\n\n\1\n\n", full_processed_text)
        norm_text = re.sub(r"(?m)^((?:Figure|Fig\.|Table)\s+\d+.*)$", r"\n\n\1\n\n", norm_text, flags=re.IGNORECASE)
        norm_text = re.sub(r"\n{3,}", "\n\n", norm_text)
        blocks = norm_text.split("\n\n")

        # Scan for headings to construct Table of Contents
        detected_headings: List[Tuple[int, str]] = []
        for block in blocks:
            c = block.strip()
            hm = re.match(r"^(#{1,4})\s+(.*)$", c)
            if hm:
                lvl = len(hm.group(1))
                txt = hm.group(2).strip()
                detected_headings.append((lvl, txt))

        # --- 4. Insert Professional Table of Contents (TOC) ---
        if include_toc and len(detected_headings) >= 2:
            # TOC Header
            toc_p = new_doc.add_paragraph()
            toc_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            toc_run = toc_p.add_run("Table of Contents")
            toc_run.font.name = typography_preset
            toc_run.font.size = Pt(18)
            toc_run.font.bold = True
            toc_run.font.color.rgb = RGBColor(26, 35, 126)
            toc_p.paragraph_format.space_after = Pt(14)

            # Native Word Dynamic TOC XML Field
            field_p = new_doc.add_paragraph()
            field_run = field_p.add_run()
            fld1 = OxmlElement('w:fldChar')
            fld1.set(qn('w:fldCharType'), 'begin')
            instr = OxmlElement('w:instrText')
            instr.set(qn('xml:space'), 'preserve')
            instr.text = r'TOC \o "1-3" \h \z \u'
            fld2 = OxmlElement('w:fldChar')
            fld2.set(qn('w:fldCharType'), 'separate')
            fld3 = OxmlElement('w:fldChar')
            fld3.set(qn('w:fldCharType'), 'end')
            field_run._r.append(fld1)
            field_run._r.append(instr)
            field_run._r.append(fld2)
            field_run._r.append(fld3)

            # Pre-computed leader-dots rows for instant readability across all viewers
            est_page = 3 if include_title_page else 2
            for lvl, h_text in detected_headings:
                entry_p = new_doc.add_paragraph()
                indent_prefix = "    " * (lvl - 1)
                entry_p.paragraph_format.space_before = Pt(2)
                entry_p.paragraph_format.space_after = Pt(2)

                # Title
                r_t = entry_p.add_run(f"{indent_prefix}{h_text} ")
                r_t.font.name = typography_preset
                r_t.font.size = Pt(11 if lvl == 1 else 10)
                if lvl == 1:
                    r_t.font.bold = True
                    r_t.font.color.rgb = RGBColor(33, 33, 33)

                # Dotted leader
                dots_count = max(6, 62 - len(indent_prefix) * 2 - len(h_text))
                r_dots = entry_p.add_run(f" {'.' * dots_count} ")
                r_dots.font.name = typography_preset
                r_dots.font.size = Pt(9.5)
                r_dots.font.color.rgb = RGBColor(160, 160, 160)

                # Page number
                r_num = entry_p.add_run(f"Page {est_page}")
                r_num.font.name = typography_preset
                r_num.font.size = Pt(9.5)
                r_num.font.color.rgb = RGBColor(100, 100, 100)

                if lvl == 1:
                    est_page += 1

            # Clean Page Break separating TOC from Body
            new_doc.add_page_break()

        # --- 5. Render Main Content Blocks ---
        for block in blocks:
            clean_text = block.strip()
            if not clean_text:
                continue

            # Markdown heading (# Heading)
            heading_match = re.match(r"^(#{1,6})\s+(.*)$", clean_text)
            if heading_match:
                level = len(heading_match.group(1))
                h_text = heading_match.group(2).strip()
                h_para = new_doc.add_heading(h_text, level=min(level, 4))
                h_para.paragraph_format.keep_with_next = True
                continue

            # Markdown table
            if clean_text.startswith("|") and "\n|" in clean_text:
                self._add_markdown_table_to_doc(new_doc, clean_text, typography_preset)
                continue

            # Figure caption styling (centered, italic, below figure)
            if FIGURE_CAPTION_RE.match(clean_text):
                p_fig = new_doc.add_paragraph()
                p_fig.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p_fig.paragraph_format.space_before = Pt(6)
                p_fig.paragraph_format.space_after = Pt(14)
                r_fig = p_fig.add_run(clean_text)
                r_fig.font.name = typography_preset
                r_fig.font.size = Pt(10)
                r_fig.font.italic = True
                r_fig.font.color.rgb = RGBColor(97, 97, 97)
                continue

            # Table caption styling (bold title above table, keep_with_next)
            if TABLE_CAPTION_RE.match(clean_text):
                p_tbl = new_doc.add_paragraph()
                p_tbl.alignment = WD_ALIGN_PARAGRAPH.LEFT
                p_tbl.paragraph_format.space_before = Pt(14)
                p_tbl.paragraph_format.space_after = Pt(4)
                p_tbl.paragraph_format.keep_with_next = True
                r_tbl = p_tbl.add_run(clean_text)
                r_tbl.font.name = typography_preset
                r_tbl.font.size = Pt(10.5)
                r_tbl.font.bold = True
                r_tbl.font.color.rgb = RGBColor(26, 35, 126)
                continue

            # List items
            lines = clean_text.split("\n")
            if any(l.strip().startswith(("- ", "* ", "• ")) for l in lines):
                for line in lines:
                    l_clean = line.strip()
                    if l_clean.startswith(("- ", "* ", "• ")):
                        item_text = re.sub(r"^[-*•]\s*", "", l_clean)
                        lp = new_doc.add_paragraph(item_text, style="List Bullet")
                        lp.paragraph_format.line_spacing = line_spacing
                        lp.paragraph_format.space_after = Pt(3)
                        lp.alignment = chosen_align
                    elif l_clean:
                        lp = new_doc.add_paragraph(l_clean)
                        lp.paragraph_format.line_spacing = line_spacing
                        lp.paragraph_format.space_after = Pt(6)
                        lp.alignment = chosen_align
                continue

            # Standard paragraph with full academic alignment
            p = new_doc.add_paragraph(clean_text)
            p.alignment = chosen_align
            p.paragraph_format.line_spacing = line_spacing
            p.paragraph_format.space_after = Pt(6)

        # Layer C: Strip container metadata to defeat provenance inspection
        self.strip_document_metadata(new_doc)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        new_doc.save(output_path)
        return output_path

    @staticmethod
    def strip_document_metadata(doc: DocxDocument) -> None:
        """
        Erases core document properties (author, revision, title, generator).
        """
        try:
            core_props = doc.core_properties
            core_props.author = ""
            core_props.last_modified_by = ""
            core_props.title = ""
            core_props.subject = ""
            core_props.keywords = ""
            core_props.comments = ""
            core_props.category = ""
        except Exception:
            pass

    @staticmethod
    def _assemble_elements_text(elements: List[Dict[str, Any]]) -> str:
        text_blocks: List[str] = []
        for el in elements:
            if el["type"] == "heading":
                prefix = "#" * max(1, min(el.get("level", 1), 6))
                text_blocks.append(f"{prefix} {el['text']}")
            elif el["type"] == "list_item":
                text_blocks.append(f"- {el['text'].lstrip('-*• ')}")
            else:
                text_blocks.append(el["text"])
        return "\n\n".join(text_blocks)

    @staticmethod
    def _extract_tail_sentence(text: str) -> str:
        # Ignore markdown headings, table lines, or bullets when getting tail sentence for stitching
        lines = [l.strip() for l in text.split("\n") if l.strip() and not l.strip().startswith(("#", "|"))]
        if not lines:
            return ""
        sents = [s.strip() for s in re.split(r"[.!?]+", " ".join(lines)) if s.strip()]
        return sents[-1] if sents else ""

    @staticmethod
    def _add_markdown_table_to_doc(doc: DocxDocument, md_table_text: str, typography_preset: str = "Times New Roman") -> None:
        lines = [l.strip() for l in md_table_text.strip().split("\n") if l.strip().startswith("|")]
        if len(lines) < 2:
            doc.add_paragraph(md_table_text)
            return

        # Parse rows
        parsed_rows: List[List[str]] = []
        for line in lines:
            # Skip divider line like |---|---|
            if re.match(r"^\|[\s\-:|]+\|$", line):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            parsed_rows.append(cells)

        if not parsed_rows:
            return

        num_cols = max(len(r) for r in parsed_rows)
        num_rows = len(parsed_rows)

        tbl = doc.add_table(rows=num_rows, cols=num_cols)
        tbl.alignment = WD_TABLE_ALIGNMENT.CENTER
        tbl.style = "Table Grid"

        for r_idx, row_data in enumerate(parsed_rows):
            for c_idx, cell_value in enumerate(row_data):
                if c_idx < num_cols:
                    cell = tbl.cell(r_idx, c_idx)
                    cell.text = cell_value

                    # Header Row: shaded background and bold font
                    if r_idx == 0:
                        try:
                            shd = parse_xml(r'<w:shd {} w:fill="E8EEF5"/>'.format(nsdecls('w')))
                            cell._tc.get_or_add_tcPr().append(shd)
                        except Exception:
                            pass
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.name = typography_preset
                                run.font.bold = True
                                run.font.size = Pt(10.5)
                    else:
                        for paragraph in cell.paragraphs:
                            for run in paragraph.runs:
                                run.font.name = typography_preset
                                run.font.size = Pt(10)


def analyze_document_structure(text_or_chunks: Union[str, List[str], List[Dict[str, Any]]]) -> Dict[str, Any]:
    """
    Step 1 Pre-Analysis: Scans complete document to detect structural hierarchy,
    headings, tables, figures, captions, word count, and reading metrics before formatting.
    """
    processed_texts: List[str] = []
    if isinstance(text_or_chunks, str):
        processed_texts = [text_or_chunks]
    else:
        for ch in text_or_chunks:
            if isinstance(ch, dict):
                processed_texts.append(ch.get("humanized_text") or ch.get("text") or "")
            elif isinstance(ch, str):
                processed_texts.append(ch)

    full_text = "\n\n".join(processed_texts)
    # Ensure headings and figure/table captions are treated as individual blocks
    norm_text = re.sub(r"(?m)^(#{1,6}\s+.*)$", r"\n\n\1\n\n", full_text)
    norm_text = re.sub(r"(?m)^((?:Figure|Fig\.|Table)\s+\d+.*)$", r"\n\n\1\n\n", norm_text, flags=re.IGNORECASE)
    norm_text = re.sub(r"\n{3,}", "\n\n", norm_text)

    words = len(norm_text.split())
    blocks = [b.strip() for b in norm_text.split("\n\n") if b.strip()]

    headings: List[Tuple[int, str]] = []
    figure_captions: List[str] = []
    table_captions: List[str] = []
    markdown_tables_count = 0

    for b in blocks:
        hm = re.match(r"^(#{1,4})\s+(.*)$", b)
        if hm:
            headings.append((len(hm.group(1)), hm.group(2).strip()))
            continue
        if FIGURE_CAPTION_RE.match(b):
            figure_captions.append(b)
            continue
        if TABLE_CAPTION_RE.match(b):
            table_captions.append(b)
            continue
        if b.startswith("|") and "\n|" in b:
            markdown_tables_count += 1

    est_pages = max(1, round(words / 450, 1))
    reading_time_min = max(1, round(words / 220))

    return {
        "total_words": words,
        "total_blocks": len(blocks),
        "headings": headings,
        "heading_count": len(headings),
        "figure_captions": figure_captions,
        "figure_count": len(figure_captions),
        "table_captions": table_captions,
        "table_caption_count": len(table_captions),
        "markdown_tables_count": markdown_tables_count,
        "estimated_pages": est_pages,
        "reading_time_min": reading_time_min,
    }


def format_document_directly(
    text_or_chunks: Union[str, List[str], List[Dict[str, Any]]],
    output_path: str,
    include_title_page: bool = False,
    title_page_data: Optional[Dict[str, str]] = None,
    include_toc: bool = True,
    typography_preset: str = "Times New Roman",
    margins_inches: float = 1.0,
    line_spacing: float = 1.5,
    running_head: str = "Writing Enhancer Pro — Academic Manuscript",
    alignment: str = "JUSTIFY",
) -> str:
    """
    Directly formats any document content with academic typography, Title Page,
    Table of Contents, 1-inch margins, running headers, full paragraph justification,
    and dynamic page numbers WITHOUT requiring an LLM or API tokens (0 API Cost, instant execution).
    """
    chunker = DocxChunker()
    chunks = [text_or_chunks] if isinstance(text_or_chunks, str) else text_or_chunks
    return chunker.reassemble_docx(
        original_docx_path=output_path,
        humanized_chunks=chunks,
        output_path=output_path,
        include_title_page=include_title_page,
        title_page_data=title_page_data,
        include_toc=include_toc,
        typography_preset=typography_preset,
        margins_inches=margins_inches,
        line_spacing=line_spacing,
        running_head=running_head,
        alignment=alignment,
    )


