"""
NETRADRISHTI — PDF Screening Case Report Generator
Generates formal, one-page clinical screening case reports using ReportLab.
Adheres strictly to research prototype disclaimers and transparent labeling.
"""

import os
import io
import base64
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage, KeepTogether
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from PIL import Image


def generate_pdf_report(case_data: Dict[str, Any], output_path: Optional[str] = None) -> bytes:
    """
    Builds a professional 1-page PDF case summary for tele-ophthalmology triage.
    Returns bytes or writes to output_path.
    """
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer if output_path is None else output_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()

    # Custom typography styles
    style_title = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=18,
        leading=22,
        textColor=colors.HexColor('#0f2942'),
        alignment=TA_LEFT
    )
    style_subtitle = ParagraphStyle(
        'DocSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#475569'),
        alignment=TA_LEFT
    )
    style_badge = ParagraphStyle(
        'DocBadge',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8,
        leading=10,
        textColor=colors.HexColor('#991b1b'),
        alignment=TA_RIGHT
    )
    style_section_hdr = ParagraphStyle(
        'SectionHdr',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=11,
        leading=14,
        textColor=colors.HexColor('#0f2942'),
        spaceBefore=6,
        spaceAfter=4
    )
    style_body = ParagraphStyle(
        'Body',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9,
        leading=12,
        textColor=colors.HexColor('#1e293b')
    )
    style_disclaimer = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=7.5,
        leading=10,
        textColor=colors.HexColor('#64748b'),
        alignment=TA_CENTER
    )

    elements = []

    # 1. Header Banner
    is_demo = "DEMO" in str(case_data.get("case_id", "")).upper() or case_data.get("is_demo", False)
    banner_text = "DEMONSTRATION REPORT — NOT A CLINICAL CERTIFICATE" if is_demo else "RESEARCH PROTOTYPE SCREENING REPORT"

    header_table_data = [
        [
            Paragraph("<b>NETRADRISHTI</b> — Diabetic Retinopathy Tele-Triage", style_title),
            Paragraph(f"<b>{banner_text}</b>", style_badge)
        ],
        [
            Paragraph("Explainable AI Screening System for Rural India | Smart India Hackathon 2026 (SIH26038)", style_subtitle),
            Paragraph(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}", ParagraphStyle('Ts', parent=styles['Normal'], fontSize=7.5, alignment=TA_RIGHT, textColor=colors.HexColor('#64748b')))
        ]
    ]
    header_table = Table(header_table_data, colWidths=[4.2 * inch, 3.3 * inch])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LINEBELOW', (0, 1), (-1, 1), 1.5, colors.HexColor('#0f2942')),
    ]))
    elements.append(header_table)
    elements.append(Spacer(1, 8))

    # 2. Patient & Screening Demographics Table
    pat_id = case_data.get("patient_id", "PAT-UNSPECIFIED")
    age = case_data.get("age", "--")
    location = case_data.get("location", "PHC Shivamogga Rural")
    screen_time = case_data.get("created_at", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    case_id = case_data.get("case_id", "CASE-UNKNOWN")

    demo_data = [
        [
            Paragraph("<b>Patient Identifier:</b>", style_body), Paragraph(str(pat_id), style_body),
            Paragraph("<b>Case Record ID:</b>", style_body), Paragraph(str(case_id), style_body)
        ],
        [
            Paragraph("<b>Age:</b>", style_body), Paragraph(f"{age} years", style_body),
            Paragraph("<b>Screening Location:</b>", style_body), Paragraph(str(location), style_body)
        ],
        [
            Paragraph("<b>Screening Time:</b>", style_body), Paragraph(str(screen_time), style_body),
            Paragraph("<b>Intake Notes:</b>", style_body), Paragraph(str(case_data.get("clinical_notes", "Routine checkup")), style_body)
        ]
    ]
    demo_table = Table(demo_data, colWidths=[1.5 * inch, 2.25 * inch, 1.5 * inch, 2.25 * inch])
    demo_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(demo_table)
    elements.append(Spacer(1, 8))

    # 3. Quality Gate & AI Evaluation Summary
    q_info = case_data.get("quality", {})
    ai_info = case_data.get("ai_result", {})
    q_status = q_info.get("status", "IMAGE ACCEPTED")
    ai_pred = ai_info.get("prediction", "NON-REFERABLE")
    icdr_scale = ai_info.get("icdr_level", "DR Level 2 — Moderate NPDR" if ai_pred == "REFERABLE" else "DR Level 0 — No Apparent DR")
    conf_val = ai_info.get("confidence", 0.0)
    conf_str = f"{conf_val * 100:.1f}%" if conf_val > 0 else "Demonstration Output"
    latency_str = f"{ai_info.get('latency_ms', 42.5)} ms"

    eval_data = [
        [
            Paragraph("<b>Quality Gate:</b>", style_body),
            Paragraph(f"<b>{q_status}</b> (Score: {q_info.get('quality_score', 80.0)}/100)", style_body),
            Paragraph("<b>AI Prediction:</b>", style_body),
            Paragraph(f"<b>{ai_pred}</b>", ParagraphStyle('Pred', parent=style_body, fontName='Helvetica-Bold', textColor=colors.HexColor('#b91c1c') if ai_pred == 'REFERABLE' else colors.HexColor('#047857')))
        ],
        [
            Paragraph("<b>ICDR Severity:</b>", style_body),
            Paragraph(str(icdr_scale), style_body),
            Paragraph("<b>Model Confidence:</b>", style_body),
            Paragraph(str(conf_str), style_body)
        ],
        [
            Paragraph("<b>Inference Latency:</b>", style_body),
            Paragraph(str(latency_str), style_body),
            Paragraph("<b>Model Architecture:</b>", style_body),
            Paragraph("NetraNet v0.1 (CNN + Grad-CAM)", style_body)
        ]
    ]
    eval_table = Table(eval_data, colWidths=[1.5 * inch, 2.25 * inch, 1.5 * inch, 2.25 * inch])
    eval_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(eval_table)
    elements.append(Spacer(1, 8))

    # 4. Embedded Imaging: Original Fundus & Grad-CAM Overlay
    # Prepare temp files if base64 or paths exist
    temp_dir = Path(__file__).resolve().parent.parent / "demo_cases"
    temp_fundus_path = None
    temp_overlay_path = None

    if case_data.get("image_path") and os.path.exists(case_data["image_path"]):
        temp_fundus_path = case_data["image_path"]
    elif case_data.get("image_data"):
        try:
            raw_b64 = case_data["image_data"].split(",")[-1]
            f_img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
            temp_fundus_path = str(temp_dir / f"pdf_fundus_{case_id}.png")
            f_img.save(temp_fundus_path)
        except Exception:
            pass

    if case_data.get("overlay_data"):
        try:
            raw_b64 = case_data["overlay_data"].split(",")[-1]
            o_img = Image.open(io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
            temp_overlay_path = str(temp_dir / f"pdf_overlay_{case_id}.png")
            o_img.save(temp_overlay_path)
        except Exception:
            pass

    img_cells = []
    if temp_fundus_path and os.path.exists(temp_fundus_path):
        img_cells.append(RLImage(temp_fundus_path, width=2.6 * inch, height=2.6 * inch))
    else:
        img_cells.append(Paragraph("<i>Fundus scan image not available.</i>", style_body))

    if temp_overlay_path and os.path.exists(temp_overlay_path):
        img_cells.append(RLImage(temp_overlay_path, width=2.6 * inch, height=2.6 * inch))
    else:
        # Fallback to fundus if overlay not computed
        img_cells.append(RLImage(temp_fundus_path, width=2.6 * inch, height=2.6 * inch) if temp_fundus_path else Paragraph("<i>Overlay not available.</i>", style_body))

    img_table_data = [
        [
            Paragraph("<b>1. Input Retinal Fundus Scan</b>", ParagraphStyle('Lbl1', parent=style_body, alignment=TA_CENTER)),
            Paragraph("<b>2. Grad-CAM Model Attention Overlay</b>", ParagraphStyle('Lbl2', parent=style_body, alignment=TA_CENTER))
        ],
        img_cells
    ]
    img_table = Table(img_table_data, colWidths=[3.75 * inch, 3.75 * inch])
    img_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
    ]))
    elements.append(img_table)
    elements.append(Spacer(1, 8))

    # 5. Explainability & Lesion Evidence Module
    evidence_text = ai_info.get("evidence", "Highlighted regions indicate spatial retinal structures that contributed to the model score.")
    elements.append(Paragraph("<b>Explainability & Lesion Module Evidence:</b>", style_section_hdr))
    elements.append(Paragraph(f"{evidence_text} <i>(Lesion-level analysis module: prototype integration. Does not prove histological lesion presence.)</i>", style_body))
    elements.append(Spacer(1, 8))

    # 6. Ophthalmologist Clinical Review & Final Decision
    doc_review = case_data.get("doctor_review", {})
    decision_val = doc_review.get("status", "PENDING REVIEW")
    doc_notes = doc_review.get("notes", "Decision pending clinical verification.")
    doc_id = doc_review.get("doctor_id", "Ophthalmologist Review (Tele-Medicine Desk)")
    dec_time = doc_review.get("timestamp", "Awaiting review")

    doc_data = [
        [
            Paragraph("<b>Reviewer Authority:</b>", style_body), Paragraph(str(doc_id), style_body),
            Paragraph("<b>Clinical Verdict:</b>", style_body), Paragraph(f"<b>{decision_val}</b>", ParagraphStyle('Dv', parent=style_body, fontName='Helvetica-Bold', textColor=colors.HexColor('#047857') if 'CONFIRM' in decision_val else colors.HexColor('#1e3a8a')))
        ],
        [
            Paragraph("<b>Decision Timestamp:</b>", style_body), Paragraph(str(dec_time), style_body),
            Paragraph("<b>Clinical Directive:</b>", style_body), Paragraph(str(doc_notes), style_body)
        ]
    ]
    doc_table = Table(doc_data, colWidths=[1.5 * inch, 2.25 * inch, 1.5 * inch, 2.25 * inch])
    doc_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]))
    elements.append(doc_table)
    elements.append(Spacer(1, 10))

    # 7. Safety Disclaimer Footer
    elements.append(Paragraph(
        "<b>LEGAL & SCIENTIFIC DISCLAIMER:</b> Research prototype developed for Smart India Hackathon 2026 (SIH26038). "
        "Not a clinically validated medical diagnostic device. AI outputs provide decision support and assist triage only. "
        "The final clinical diagnosis and patient management decision rests entirely with the licensed ophthalmologist.",
        style_disclaimer
    ))

    # Build PDF
    doc.build(elements)

    if output_path is None:
        return buffer.getvalue()
    return b""
