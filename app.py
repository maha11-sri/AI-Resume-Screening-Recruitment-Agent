import streamlit as st
from agent.recruitment_agent import recruitment_agent
import pandas as pd
from io import BytesIO
from html import escape

from parser.resume_parser import read_pdf, read_docx, read_txt
from preprocessing.text_preprocessing import clean_text
from models.similarity import calculate_similarity
from models.semantic_similarity import calculate_semantic_similarity
from models.skill_matcher import calculate_skill_match
from models.candidate_info import extract_candidate_info


# ============================================================
# PDF IMPORTS
# ============================================================

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    PageBreak
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Resume Screening System",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# SESSION STATE
# ============================================================

if "recruiter_decisions" not in st.session_state:
    st.session_state.recruiter_decisions = {}

if "agent_chat_history" not in st.session_state:
    st.session_state.agent_chat_history = []


# ============================================================
# RECRUITER DECISION FUNCTIONS
# ============================================================

def get_recruiter_status(filename):
    return st.session_state.recruiter_decisions.get(
        filename,
        "Under Review"
    )


def set_recruiter_status(filename, status):
    st.session_state.recruiter_decisions[filename] = status


def status_display(status):
    if status == "Shortlisted":
        return "🟢 Shortlisted"

    if status == "Rejected":
        return "🔴 Rejected"

    return "🟡 Under Review"
st.markdown(
    """
    <style>

    div[data-testid="stDialog"] > div {
        border-radius: 20px !important;
    }

    div[data-testid="stDialog"] section {
        border-radius: 20px !important;
    }
    div[data-testid="stDialog"] button[aria-label="Close"] {
        color: red !important;
    }

    div[data-testid="stDialog"] button[aria-label="Close"]:hover {
        color: darkred !important;
        background-color: #ffe5e5 !important;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# QUICK OVERVIEW DIALOG
# ============================================================

@st.dialog("🏆 Top Performance Overview")
def show_quick_overview(ranked_results):

    # Popup styling
     # Show only top 3 candidates
    st.subheader("🏆 Top Performance Overview")

    st.caption(
        "Top candidates ranked by Final Match Score"
    )
    

    top_candidates = ranked_results[:3]

    medals = ["🥇", "🥈", "🥉"]

    for index, candidate in enumerate(top_candidates):

        st.markdown(
            f"### {medals[index]} Top {index + 1}"
        )

        with st.container(border=True):

            st.write(
                f"👤 **Candidate:** "
                f"{candidate['name']}"
            )

            st.write(
                f"🎯 **Final Match Score:** "
                f"{candidate['final_score']:.2f}%"
            )

            skills = candidate.get(
                "candidate_skills",
                []
            )

            top_skills = ", ".join(
                skills[:6]
            )

            if not top_skills:
                top_skills = "No skills found"

            st.write(
                f"🛠️ **Top Skills:** "
                f"{top_skills}"
            )

# ============================================================
# PDF SAFE TEXT
# ============================================================

def pdf_safe_text(value):
    if value is None:
        return ""

    text = str(value)

    replacements = {
        "🟢 Excellent Match": "[EXCELLENT MATCH]",
        "🟡 Moderate Match": "[MODERATE MATCH]",
        "🔴 Low Match": "[LOW MATCH]",

        "🟢 Shortlisted": "[SHORTLISTED]",
        "🔴 Rejected": "[REJECTED]",
        "🟡 Under Review": "[UNDER REVIEW]",

        "✓": "OK",
        "✔": "OK",
        "✗": "X",
        "❌": "X",
        "✅": "OK",

        "⚠️": "WARNING",
        "⚠": "WARNING",

        "💡": "Recommendation:",

        "📄": "",
        "👤": "",
        "📊": "",
        "🛠️": "",
        "🧠": "",
        "🤖": "",
        "🎯": "",
        "📌": "",
        "🏆": "",
        "📈": "",
        "📋": "",
        "🧑‍💼": "",
        "📧": "",
        "📞": "",
        "🎓": "",
        "👥": "",
        "🔎": "",
        "🔽": "",
        "🎉": "",
        "🥇": "",
        "🥈": "",
        "🥉": ""
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return escape(text)


# ============================================================
# AI RECOMMENDATION
# ============================================================

def get_recommendation(skill_coverage, final_score):

    if skill_coverage >= 70 and final_score >= 50:
        return "🟢 Excellent Match"

    if skill_coverage >= 40 and final_score >= 30:
        return "🟡 Moderate Match"

    return "🔴 Low Match"


# ============================================================
# AI RECOMMENDATION DESCRIPTION
# ============================================================

def get_recommendation_description(
    skill_coverage,
    final_score
):

    if skill_coverage >= 70 and final_score >= 50:
        return (
            "Excellent Match. The candidate demonstrates "
            "strong coverage of the required job skills "
            "and strong overall screening alignment."
        )

    if skill_coverage >= 40 and final_score >= 30:
        return (
            "Moderate Match. The candidate covers a reasonable "
            "portion of the required skills but may require "
            "additional technical evaluation."
        )

    return (
        "Low Match. The candidate currently shows limited "
        "coverage of the required job skills or limited "
        "overall alignment with the job description."
    )


# ============================================================
# SKILL COVERAGE EXPLANATION
# ============================================================

def generate_skill_coverage_explanation(result):

    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])
    coverage = float(result.get("skill_coverage", 0))

    total_required = len(matched) + len(missing)

    if total_required == 0:
        return (
            "Skill Coverage could not be calculated because "
            "no required skills were identified from the "
            "job description."
        )

    if coverage >= 80:
        level = "Very strong"
    elif coverage >= 60:
        level = "Strong"
    elif coverage >= 40:
        level = "Moderate"
    elif coverage > 0:
        level = "Low"
    else:
        level = "Very low"

    explanation = (
        f"{level} skill coverage: the candidate matches "
        f"{len(matched)} out of {total_required} required "
        f"skill(s), resulting in {coverage:.2f}% skill coverage."
    )

    if matched:
        explanation += (
            f" Matched skills include: {', '.join(matched)}."
        )

    if missing:
        explanation += (
            f" Missing skills include: {', '.join(missing)}."
        )
    else:
        explanation += (
            " No required skills are currently missing."
        )

    return explanation


# ============================================================
# AI SCREENING EXPLANATION
# ============================================================

def generate_ai_explanation(result):

    matched = result.get("matched_skills", [])
    missing = result.get("missing_skills", [])

    skill_score = float(result.get("skill_match", 0))
    skill_coverage = float(result.get("skill_coverage", 0))

    semantic = float(result.get("semantic", 0))
    similarity = float(result.get("similarity", 0))

    cgpa = result.get("cgpa", "Not found")
    final_score = float(result.get("final_score", 0))

    explanation = []

    # --------------------------------------------------------
    # MATCHED SKILLS
    # --------------------------------------------------------

    if matched:
        explanation.append(
            f"OK {len(matched)} required skill(s) matched: "
            f"{', '.join(matched)}"
        )
    else:
        explanation.append(
            "X No required skills were matched."
        )

    # --------------------------------------------------------
    # MISSING SKILLS
    # --------------------------------------------------------

    if missing:
        explanation.append(
            f"X {len(missing)} required skill(s) missing: "
            f"{', '.join(missing)}"
        )
    else:
        explanation.append(
            "OK No required skills are missing."
        )

    # --------------------------------------------------------
    # SKILL MATCH
    # --------------------------------------------------------

    if skill_score >= 70:
        explanation.append(
            "OK Strong technical skill matching "
            "with the job description."
        )
    elif skill_score >= 40:
        explanation.append(
            "WARNING Moderate technical skill matching "
            "with the job description."
        )
    else:
        explanation.append(
            "X Weak technical skill matching "
            "with the job description."
        )

    # --------------------------------------------------------
    # SKILL COVERAGE
    # --------------------------------------------------------

    coverage_text = generate_skill_coverage_explanation(result)

    explanation.append(
        f"Skill Coverage Analysis: {coverage_text}"
    )

    # --------------------------------------------------------
    # SEMANTIC SIMILARITY
    # --------------------------------------------------------

    if semantic >= 60:
        explanation.append(
            "OK Strong semantic similarity between "
            "the resume and job description."
        )
    elif semantic >= 30:
        explanation.append(
            "WARNING Moderate semantic similarity between "
            "the resume and job description."
        )
    else:
        explanation.append(
            "X Low semantic similarity between "
            "the resume and job description."
        )

    # --------------------------------------------------------
    # TEXT SIMILARITY
    # --------------------------------------------------------

    if similarity >= 60:
        explanation.append(
            "OK Strong keyword/text-level alignment."
        )
    elif similarity >= 30:
        explanation.append(
            "WARNING Moderate keyword/text-level alignment."
        )
    else:
        explanation.append(
            "X Low keyword/text-level alignment."
        )

    # --------------------------------------------------------
    # CGPA
    # --------------------------------------------------------

    try:
        cgpa_value = float(cgpa)

        if cgpa_value >= 8.5:
            explanation.append(
                f"OK Strong academic performance "
                f"with CGPA {cgpa_value:.2f}."
            )
        elif cgpa_value >= 7.0:
            explanation.append(
                f"OK Acceptable academic performance "
                f"with CGPA {cgpa_value:.2f}."
            )
        else:
            explanation.append(
                f"WARNING Academic performance is relatively "
                f"low with CGPA {cgpa_value:.2f}."
            )

    except (ValueError, TypeError):
        pass

    # --------------------------------------------------------
    # FINAL SCORE
    # --------------------------------------------------------

    explanation.append(
        f"Final screening score is {final_score:.2f}%."
    )

    # --------------------------------------------------------
    # RECOMMENDATION DESCRIPTION
    # --------------------------------------------------------

    recommendation_description = get_recommendation_description(
        skill_coverage,
        final_score
    )

    explanation.append(
        f"Recommendation: {recommendation_description}"
    )

    # --------------------------------------------------------
    # IMPORTANT MESSAGE
    # --------------------------------------------------------

    explanation.append(
        "Important: AI recommendation is advisory. "
        "Recruiter decision is independent and must be "
        "selected explicitly by the recruiter."
    )

    return explanation


# ============================================================
# PDF REPORT GENERATION
# ============================================================

def generate_pdf_report(
    ranked_results,
    total_candidates,
    average_score,
    best_candidate,
    shortlisted_count,
    review_count,
    rejected_count
):

    buffer = BytesIO()

    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=32,
        leftMargin=32,
        topMargin=35,
        bottomMargin=35,
        title="AI Resume Screening Report",
        author="AI Resume Screening System"
    )

    styles = getSampleStyleSheet()

    # --------------------------------------------------------
    # PDF STYLES
    # --------------------------------------------------------

    title_style = ParagraphStyle(
        "PDFTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        fontName="Helvetica-Bold",
        fontSize=20,
        leading=24,
        spaceAfter=8
    )

    subtitle_style = ParagraphStyle(
        "PDFSubtitle",
        parent=styles["Normal"],
        alignment=TA_CENTER,
        fontName="Helvetica",
        fontSize=11,
        leading=14,
        spaceAfter=20
    )

    heading_style = ParagraphStyle(
        "PDFHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=14,
        leading=18,
        spaceBefore=8,
        spaceAfter=10
    )

    subheading_style = ParagraphStyle(
        "PDFSubHeading",
        parent=styles["Heading3"],
        fontName="Helvetica-Bold",
        fontSize=11,
        leading=14,
        spaceBefore=8,
        spaceAfter=6
    )

    normal_style = ParagraphStyle(
        "PDFNormal",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        spaceAfter=4
    )

    table_header_style = ParagraphStyle(
        "PDFTableHeader",
        parent=styles["BodyText"],
        fontName="Helvetica-Bold",
        fontSize=8,
        leading=10
    )

    table_style = ParagraphStyle(
        "PDFTableText",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=7.5,
        leading=10
    )

    explanation_style = ParagraphStyle(
        "PDFExplanation",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8,
        leading=11,
        spaceAfter=3
    )

    story = []

    # ========================================================
    # TITLE
    # ========================================================

    story.append(
        Paragraph(
            "AI Resume Screening System",
            title_style
        )
    )

    story.append(
        Paragraph(
            "Recruiter Screening Report",
            subtitle_style
        )
    )

    # ========================================================
    # DASHBOARD
    # ========================================================

    story.append(
        Paragraph(
            "Recruiter Dashboard",
            heading_style
        )
    )

    dashboard_data = [
        [
            Paragraph("Metric", table_header_style),
            Paragraph("Value", table_header_style)
        ],
        [
            Paragraph("Total Candidates", table_style),
            Paragraph(str(total_candidates), table_style)
        ],
        [
            Paragraph("Average Match", table_style),
            Paragraph(
                f"{average_score:.2f}%",
                table_style
            )
        ],
        [
            Paragraph("Best Candidate", table_style),
            Paragraph(
                pdf_safe_text(best_candidate["name"]),
                table_style
            )
        ],
        [
            Paragraph("Best Candidate Score", table_style),
            Paragraph(
                f"{best_candidate['final_score']:.2f}%",
                table_style
            )
        ],
        [
            Paragraph("Shortlisted", table_style),
            Paragraph(str(shortlisted_count), table_style)
        ],
        [
            Paragraph("Under Review", table_style),
            Paragraph(str(review_count), table_style)
        ],
        [
            Paragraph("Rejected", table_style),
            Paragraph(str(rejected_count), table_style)
        ]
    ]

    dashboard_table = Table(
        dashboard_data,
        colWidths=[250, 250],
        repeatRows=1
    )

    dashboard_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(dashboard_table)
    story.append(Spacer(1, 18))

    # ========================================================
    # RECRUITMENT PIPELINE
    # ========================================================

    story.append(
        Paragraph(
            "Recruitment Pipeline",
            heading_style
        )
    )

    pipeline_data = [
        [
            Paragraph("Recruiter Status", table_header_style),
            Paragraph("Candidates", table_header_style)
        ],
        [
            Paragraph("Shortlisted", table_style),
            Paragraph(str(shortlisted_count), table_style)
        ],
        [
            Paragraph("Under Review", table_style),
            Paragraph(str(review_count), table_style)
        ],
        [
            Paragraph("Rejected", table_style),
            Paragraph(str(rejected_count), table_style)
        ]
    ]

    pipeline_table = Table(
        pipeline_data,
        colWidths=[250, 250],
        repeatRows=1
    )

    pipeline_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "ALIGN",
                (1, 1),
                (1, -1),
                "CENTER"
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                7
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                6
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                6
            )
        ])
    )

    story.append(pipeline_table)

    story.append(PageBreak())

    # ========================================================
    # CANDIDATE RANKING
    # ========================================================

    story.append(
        Paragraph(
            "Candidate Ranking",
            heading_style
        )
    )

    ranking_data = [
        [
            Paragraph("Rank", table_header_style),
            Paragraph("Candidate", table_header_style),
            Paragraph("Final", table_header_style),
            Paragraph("Skills", table_header_style),
            Paragraph("Coverage", table_header_style),
            Paragraph("AI Recommendation", table_header_style),
            Paragraph("Recruiter Decision", table_header_style)
        ]
    ]

    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        recommendation = result["recommendation"]

        if "Excellent Match" in recommendation:
            recommendation = "Excellent Match"
        elif "Moderate Match" in recommendation:
            recommendation = "Moderate Match"
        elif "Low Match" in recommendation:
            recommendation = "Low Match"
        else:
            recommendation = "Not Available"

        decision = get_recruiter_status(
            result["filename"]
        )

        total_required_skills = (
            len(result["matched_skills"])
            + len(result["missing_skills"])
        )

        if total_required_skills > 0:
            skill_summary = (
                f"{len(result['matched_skills'])}/"
                f"{total_required_skills}"
            )
        else:
            skill_summary = "0/0"

        ranking_data.append([
            Paragraph(str(rank), table_style),
            Paragraph(
                pdf_safe_text(result["name"]),
                table_style
            ),
            Paragraph(
                f"{result['final_score']:.2f}%",
                table_style
            ),
            Paragraph(
                skill_summary,
                table_style
            ),
            Paragraph(
                f"{result['skill_coverage']:.2f}%",
                table_style
            ),
            Paragraph(
                pdf_safe_text(recommendation),
                table_style
            ),
            Paragraph(
                pdf_safe_text(decision),
                table_style
            )
        ])

    ranking_table = Table(
        ranking_data,
        colWidths=[
            32,
            85,
            55,
            55,
            60,
            105,
            105
        ],
        repeatRows=1
    )

    ranking_table.setStyle(
        TableStyle([
            (
                "BACKGROUND",
                (0, 0),
                (-1, 0),
                colors.lightgrey
            ),
            (
                "GRID",
                (0, 0),
                (-1, -1),
                0.5,
                colors.grey
            ),
            (
                "VALIGN",
                (0, 0),
                (-1, -1),
                "MIDDLE"
            ),
            (
                "ALIGN",
                (0, 0),
                (0, -1),
                "CENTER"
            ),
            (
                "ALIGN",
                (2, 1),
                (4, -1),
                "CENTER"
            ),
            (
                "ALIGN",
                (6, 1),
                (6, -1),
                "CENTER"
            ),
            (
                "LEFTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "RIGHTPADDING",
                (0, 0),
                (-1, -1),
                4
            ),
            (
                "TOPPADDING",
                (0, 0),
                (-1, -1),
                5
            ),
            (
                "BOTTOMPADDING",
                (0, 0),
                (-1, -1),
                5
            )
        ])
    )

    story.append(ranking_table)

    story.append(PageBreak())

    # ========================================================
    # DETAILED CANDIDATE ANALYSIS
    # ========================================================

    story.append(
        Paragraph(
            "Detailed Candidate Analysis",
            heading_style
        )
    )

    for rank, result in enumerate(
        ranked_results,
        start=1
    ):

        candidate_skills = result["candidate_skills"]
        matched_skills = result["matched_skills"]
        missing_skills = result["missing_skills"]

        explanation = generate_ai_explanation(result)

        recruiter_status = get_recruiter_status(
            result["filename"]
        )

        story.append(
            Paragraph(
                f"{rank}. {pdf_safe_text(result['name'])}",
                subheading_style
            )
        )

        candidate_data = [
            [
                Paragraph("Field", table_header_style),
                Paragraph("Details", table_header_style)
            ],
            [
                Paragraph("Email", table_style),
                Paragraph(
                    pdf_safe_text(result["email"]),
                    table_style
                )
            ],
            [
                Paragraph("Phone", table_style),
                Paragraph(
                    pdf_safe_text(result["phone"]),
                    table_style
                )
            ],
            [
                Paragraph("Education", table_style),
                Paragraph(
                    pdf_safe_text(result["education"]),
                    table_style
                )
            ],
            [
                Paragraph("CGPA", table_style),
                Paragraph(
                    pdf_safe_text(result["cgpa"]),
                    table_style
                )
            ],
            [
                Paragraph("Skills", table_style),
                Paragraph(
                    pdf_safe_text(
                        ", ".join(candidate_skills)
                        if candidate_skills
                        else "None"
                    ),
                    table_style
                )
            ],
            [
                Paragraph("Final Score", table_style),
                Paragraph(
                    f"{result['final_score']:.2f}%",
                    table_style
                )
            ],
            [
                Paragraph("Skill Match", table_style),
                Paragraph(
                    f"{result['skill_match']:.2f}%",
                    table_style
                )
            ],
            [
                Paragraph("Skill Coverage", table_style),
                Paragraph(
                    f"{result['skill_coverage']:.2f}%",
                    table_style
                )
            ],
            [
                Paragraph("Semantic Similarity", table_style),
                Paragraph(
                    f"{result['semantic']:.2f}%",
                    table_style
                )
            ],
            [
                Paragraph("Text Similarity", table_style),
                Paragraph(
                    f"{result['similarity']:.2f}%",
                    table_style
                )
            ],
            [
                Paragraph("Matched Skills", table_style),
                Paragraph(
                    pdf_safe_text(
                        ", ".join(matched_skills)
                        if matched_skills
                        else "None"
                    ),
                    table_style
                )
            ],
            [
                Paragraph("Missing Skills", table_style),
                Paragraph(
                    pdf_safe_text(
                        ", ".join(missing_skills)
                        if missing_skills
                        else "None"
                    ),
                    table_style
                )
            ],
            [
                Paragraph(
                    "AI Recommendation",
                    table_style
                ),
                Paragraph(
                    pdf_safe_text(
                        result["recommendation"]
                    ),
                    table_style
                )
            ],
            [
                Paragraph(
                    "Recruiter Decision",
                    table_style
                ),
                Paragraph(
                    pdf_safe_text(
                        recruiter_status
                    ),
                    table_style
                )
            ]
        ]

        candidate_table = Table(
            candidate_data,
            colWidths=[145, 355],
            repeatRows=1
        )

        candidate_table.setStyle(
            TableStyle([
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 0),
                    colors.lightgrey
                ),
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    0.5,
                    colors.grey
                ),
                (
                    "VALIGN",
                    (0, 0),
                    (-1, -1),
                    "TOP"
                ),
                (
                    "FONTNAME",
                    (0, 1),
                    (0, -1),
                    "Helvetica-Bold"
                ),
                (
                    "LEFTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "RIGHTPADDING",
                    (0, 0),
                    (-1, -1),
                    6
                ),
                (
                    "TOPPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                ),
                (
                    "BOTTOMPADDING",
                    (0, 0),
                    (-1, -1),
                    5
                )
            ])
        )

        story.append(candidate_table)
        story.append(Spacer(1, 10))

        story.append(
            Paragraph(
                "AI Screening Explanation",
                subheading_style
            )
        )

        for item in explanation:
            story.append(
                Paragraph(
                    pdf_safe_text(item),
                    explanation_style
                )
            )

        story.append(Spacer(1, 12))

        story.append(
            Paragraph(
                "Recruiter Decision",
                subheading_style
            )
        )

        story.append(
            Paragraph(
                f"Final recruiter decision: "
                f"{pdf_safe_text(recruiter_status)}",
                normal_style
            )
        )

        if rank < len(ranked_results):
            story.append(PageBreak())

    # ========================================================
    # RECOMMENDED CANDIDATE
    # ========================================================

    story.append(PageBreak())

    story.append(
        Paragraph(
            "Recommended Candidate",
            heading_style
        )
    )

    best_status = get_recruiter_status(
        best_candidate["filename"]
    )

    best_name = pdf_safe_text(
        best_candidate["name"]
    )

    best_score = best_candidate["final_score"]
    best_coverage = best_candidate["skill_coverage"]

    story.append(
        Paragraph(
            f"<b>{best_name}</b> is currently the "
            f"highest-ranked candidate with a final "
            f"match score of <b>{best_score:.2f}%</b>.",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"Skill Coverage: "
            f"<b>{best_coverage:.2f}%</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    best_recommendation = best_candidate["recommendation"]

    if "Excellent Match" in best_recommendation:
        best_recommendation = "Excellent Match"
    elif "Moderate Match" in best_recommendation:
        best_recommendation = "Moderate Match"
    elif "Low Match" in best_recommendation:
        best_recommendation = "Low Match"
    else:
        best_recommendation = "Not Available"

    story.append(
        Paragraph(
            f"AI Recommendation: "
            f"<b>{pdf_safe_text(best_recommendation)}</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 8))

    story.append(
        Paragraph(
            f"Recruiter Decision: "
            f"<b>{pdf_safe_text(best_status)}</b>",
            normal_style
        )
    )

    story.append(Spacer(1, 15))

    story.append(
        Paragraph(
            "Important: The AI recommendation is advisory. "
            "The recruiter makes the final hiring decision.",
            normal_style
        )
    )

    # ========================================================
    # BUILD PDF
    # ========================================================

    document.build(story)

    buffer.seek(0)

    return buffer


# ============================================================
# SIDEBAR - AGENT CONTROL & STATUS
# ============================================================

with st.sidebar:
    st.header("⚙️ Controls")
    
    # 1. Reset Application Button
    if st.button("🗑️ Reset Chat History", use_container_width=True):
        st.session_state.agent_chat_history = []
        recruitment_agent.clear_memory()
        st.rerun()
        
    st.divider()

    # 2. Candidate Filtering Option
    st.subheader("🎯 Screening Filters")
    min_score = st.slider(
        "Minimum Match Score (%)", 
        min_value=0, 
        max_value=100, 
        value=70,
        help="Filter candidates based on their match score."
    )

    st.divider()

    # 3. Quick Guide / Instructions
    st.subheader("📄 Quick Guide")
    st.markdown("""
    1. **Upload** a Job Description (`.txt`, `.pdf`, `.docx`).
    2. **Upload** one or more candidate resumes.
    3. View rankings & detailed breakdown below.
    """)
    
    st.info("💡 **Tip:** Use the interactive recruitment agent after screening to ask detailed candidate questions.")

# ============================================================
# APPLICATION TITLE
# ============================================================

st.title(
    "📄 AI Resume Screening System"
)

st.write(
    "Autonomous AI-powered resume screening, candidate ranking, "
    "recruiter analysis, and interactive recruitment agent."
)


# ============================================================
# JOB DESCRIPTION
# ============================================================

st.header(
    "📄 Upload Job Description"
)

uploaded_jd = st.file_uploader(
    "Upload Job Description (TXT, PDF, or DOCX)",
    type=["txt", "pdf", "docx"],
    key="job_description"
)

jd_text = None
cleaned_jd = None

if uploaded_jd is not None:
    jd_filename = uploaded_jd.name.lower()
    try:
        if jd_filename.endswith(".pdf"):
            jd_text = read_pdf(uploaded_jd)
        elif jd_filename.endswith(".docx"):
            jd_text = read_docx(uploaded_jd)
        else:
            jd_text = uploaded_jd.read().decode("utf-8", errors="ignore")
    except Exception as read_err:
        st.error(f"Error reading Job Description file: {read_err}")
        jd_text = ""

    if not jd_text or not jd_text.strip():
        st.error("The Job Description is empty or unreadable.")
    else:
        st.success("Job Description uploaded and parsed successfully!")
        cleaned_jd = clean_text(jd_text)

        with st.expander("📄 View Job Description"):
            st.text(jd_text)


# ============================================================
# RESUME UPLOAD
# ============================================================

st.header(
    "📂 Upload Candidate Resumes"
)

uploaded_resumes = st.file_uploader(
    "Upload one or more PDF/DOCX resumes",
    type=["pdf", "docx"],
    accept_multiple_files=True,
    key="candidate_resumes"
)


# ============================================================
# PROCESS RESUMES
# ============================================================

if (
    uploaded_resumes
    and uploaded_jd is not None
    and jd_text
    and jd_text.strip()
):

    st.header(
        "📊 Resume Screening Results"
    )

    results = []

    # ========================================================
    # PROCESS EACH RESUME
    # ========================================================

    for resume in uploaded_resumes:

        try:

            if resume.name.lower().endswith(".pdf"):
                text = read_pdf(resume)

            elif resume.name.lower().endswith(".docx"):
                text = read_docx(resume)

            else:
                text = ""

        except Exception as error:

            st.error(
                f"Error reading {resume.name}: {error}"
            )

            continue

        if not text or not text.strip():

            st.warning(
                f"Could not extract text from {resume.name}"
            )

            continue

        # ====================================================
        # CLEAN TEXT
        # ====================================================

        try:

            cleaned_text = clean_text(text)

        except Exception as error:

            st.error(
                f"Text cleaning failed for "
                f"{resume.name}: {error}"
            )

            continue

        # ====================================================
        # CANDIDATE INFORMATION
        # ====================================================

        try:

            candidate_info = extract_candidate_info(text)

        except Exception as error:

            st.warning(
                f"Candidate information extraction "
                f"failed for {resume.name}: {error}"
            )

            candidate_info = {
                "name": "Not found",
                "email": "Not found",
                "phone": "Not found",
                "education": "Not found",
                "cgpa": "Not found",
                "skills": []
            }

        # ====================================================
        # TEXT SIMILARITY
        # ====================================================

        try:

            similarity_score = calculate_similarity(
                cleaned_jd,
                cleaned_text
            )

            similarity_percentage = (
                float(similarity_score) * 100
            )

            similarity_percentage = min(
                max(similarity_percentage, 0.0),
                100.0
            )

        except Exception as error:

            st.warning(
                f"Text similarity calculation failed "
                f"for {resume.name}: {error}"
            )

            similarity_percentage = 0.0

        # ====================================================
        # SEMANTIC SIMILARITY
        # ====================================================

        try:

            semantic_score = calculate_semantic_similarity(
                jd_text,
                text
            )

            semantic_percentage = float(
                semantic_score
            )

            semantic_percentage = min(
                max(semantic_percentage, 0.0),
                100.0
            )

        except Exception as error:

            st.warning(
                f"Semantic similarity calculation failed "
                f"for {resume.name}: {error}"
            )

            semantic_percentage = 0.0

        # ====================================================
        # SKILL MATCH
        # ====================================================

        try:

            (
                skill_score,
                matched_skills,
                missing_skills,
                candidate_skills
            ) = calculate_skill_match(
                jd_text,
                text
            )

            skill_score = float(skill_score)

        except Exception as error:

            st.warning(
                f"Skill matching failed for "
                f"{resume.name}: {error}"
            )

            skill_score = 0.0
            matched_skills = []
            missing_skills = []
            candidate_skills = []

        # ====================================================
        # SKILL COVERAGE
        # ====================================================

        total_required_skills = (
            len(matched_skills)
            + len(missing_skills)
        )

        if total_required_skills > 0:

            skill_coverage = (
                len(matched_skills)
                / total_required_skills
            ) * 100

        else:

            skill_coverage = 0.0

        skill_coverage = min(
            max(
                float(skill_coverage),
                0.0
            ),
            100.0
        )

        # ====================================================
        # FINAL SCORE
        # ====================================================

        final_score = (
            (skill_score * 0.50)
            + (semantic_percentage * 0.30)
            + (similarity_percentage * 0.20)
        )

        final_score = round(
            min(
                max(final_score, 0.0),
                100.0
            ),
            2
        )

        # ====================================================
        # AI RECOMMENDATION
        # ====================================================

        recommendation = get_recommendation(
            skill_coverage,
            final_score
        )

        # ====================================================
        # STORE RESULT
        # ====================================================

        result = {
            "name": candidate_info.get(
                "name",
                "Not found"
            ),

            "filename": resume.name,

            "email": candidate_info.get(
                "email",
                "Not found"
            ),

            "phone": candidate_info.get(
                "phone",
                "Not found"
            ),

            "education": candidate_info.get(
                "education",
                "Not found"
            ),

            "cgpa": candidate_info.get(
                "cgpa",
                "Not found"
            ),

            "candidate_skills": candidate_skills,

            "similarity": similarity_percentage,

            "semantic": semantic_percentage,

            "skill_match": skill_score,

            "skill_coverage": skill_coverage,

            "final_score": final_score,

            "recommendation": recommendation,

            "matched_skills": matched_skills,

            "missing_skills": missing_skills,

            "raw_text": text,

            "cleaned_text": cleaned_text
        }

        results.append(result)

    # ========================================================
    # RECRUITER DASHBOARD RENDERER FUNCTION
    # ========================================================

    def render_recruiter_dashboard(ranked_results):

        if st.button("⚡ Quick Overview", type="primary", key="quick_overview_btn"):
            show_quick_overview(ranked_results)

        st.header(
            "📊 Recruiter Dashboard"
        )

        total_candidates = len(
            ranked_results
        )

        average_score = (
            sum(
                result["final_score"]
                for result in ranked_results
            )
            / total_candidates
        )

        best_candidate = ranked_results[0]

        # ====================================================
        # MISSING SKILLS
        # ====================================================

        missing_skill_count = {}

        for result in ranked_results:

            for skill in result["missing_skills"]:

                skill_lower = skill.lower()

                missing_skill_count[skill_lower] = (
                    missing_skill_count.get(
                        skill_lower,
                        0
                    ) + 1
                )

        if missing_skill_count:

            most_missing_skill = max(
                missing_skill_count,
                key=missing_skill_count.get
            )

        else:

            most_missing_skill = "None"

        # ====================================================
        # PIPELINE COUNTS
        # ====================================================

        shortlisted_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Shortlisted"
        )

        review_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Under Review"
        )

        rejected_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Rejected"
        )

        # ====================================================
        # DASHBOARD METRICS
        # ====================================================

        col1, col2, col3, col4 = st.columns(4)

        with col1:

            st.metric(
                "👥 Total Candidates",
                total_candidates
            )

        with col2:

            st.metric(
                "📈 Average Match",
                f"{average_score:.2f}%"
            )

        with col3:

            st.metric(
                "🏆 Best Candidate",
                best_candidate["name"]
            )

        with col4:

            st.metric(
                "🛠️ Most Missing Skill",
                most_missing_skill
            )

        st.divider()

        # ====================================================
        # RECRUITMENT PIPELINE
        # ====================================================

        st.header(
            "📌 Recruitment Pipeline"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🟢 Shortlisted",
                shortlisted_count
            )

        with col2:

            st.metric(
                "🟡 Under Review",
                review_count
            )

        with col3:

            st.metric(
                "🔴 Rejected",
                rejected_count
            )

        st.divider()

        # ====================================================
        # CANDIDATE RANKING
        # ====================================================

        st.header(
            "🏆 Candidate Ranking"
        )

        for rank, result in enumerate(
            ranked_results,
            start=1
        ):

            if rank == 1:
                medal = "🥇"
            elif rank == 2:
                medal = "🥈"
            elif rank == 3:
                medal = "🥉"
            else:
                medal = f"{rank}."

            total_required_skills = (
                len(result["matched_skills"])
                + len(result["missing_skills"])
            )

            if total_required_skills > 0:

                skill_summary = (
                    f"{len(result['matched_skills'])}/"
                    f"{total_required_skills}"
                )

            else:

                skill_summary = "0/0"

            current_status = get_recruiter_status(
                result["filename"]
            )

            st.write(
                f"{medal} "
                f"**{result['name']}** — "
                f"**{result['final_score']:.2f}%** — "
                f"**{result['recommendation']}**"
            )

            st.caption(
                f"🛠️ Skills Matched: {skill_summary}"
            )

            st.caption(
                f"📊 Skill Coverage: "
                f"{result['skill_coverage']:.2f}%"
            )

            st.caption(
                f"📌 Recruiter Decision: "
                f"{status_display(current_status)}"
            )

        st.divider()

        # ====================================================
        # CANDIDATE COMPARISON
        # ====================================================

        st.header(
            "📊 Candidate Comparison"
        )

        for result in ranked_results:

            st.subheader(
                f"👤 {result['name']}"
            )

            st.write(
                f"🎯 Final Match Score: "
                f"**{result['final_score']:.2f}%**"
            )

            st.progress(
                min(
                    max(
                        result["final_score"] / 100,
                        0.0
                    ),
                    1.0
                )
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "🎯 Final Score",
                    f"{result['final_score']:.2f}%"
                )

            with col2:

                st.metric(
                    "🛠️ Skill Match",
                    f"{result['skill_match']:.2f}%"
                )

            with col3:

                st.metric(
                    "📊 Skill Coverage",
                    f"{result['skill_coverage']:.2f}%"
                )

            with col4:

                st.metric(
                    "🧠 Semantic Similarity",
                    f"{result['semantic']:.2f}%"
                )

            st.write(
                f"🤖 Text Similarity: "
                f"{result['similarity']:.2f}%"
            )

            if result["matched_skills"]:

                st.write(
                    "✅ Matched Skills: "
                    + ", ".join(
                        result["matched_skills"]
                    )
                )

            else:

                st.write(
                    "✅ Matched Skills: None"
                )

            if result["missing_skills"]:

                st.write(
                    "❌ Missing Skills: "
                    + ", ".join(
                        result["missing_skills"]
                    )
                )

            else:

                st.write(
                    "🎉 Missing Skills: None"
                )

            st.write(
                f"📌 AI Recommendation: "
                f"**{result['recommendation']}**"
            )

            current_status = get_recruiter_status(
                result["filename"]
            )

            st.write(
                f"🧑‍💼 Recruiter Decision: "
                f"**{status_display(current_status)}**"
            )

            st.markdown(
                "### 🤖 AI Screening Explanation"
            )

            explanation = generate_ai_explanation(
                result
            )

            for item in explanation:
                st.write(item)

            st.divider()

        # ====================================================
        # CANDIDATE SUMMARY
        # ====================================================

        st.header(
            "📋 Candidate Summary"
        )

        summary_data = []

        for rank, result in enumerate(
            ranked_results,
            start=1
        ):

            total_required_skills = (
                len(result["matched_skills"])
                + len(result["missing_skills"])
            )

            if total_required_skills > 0:

                skill_summary = (
                    f"{len(result['matched_skills'])}/"
                    f"{total_required_skills}"
                )

            else:

                skill_summary = "0/0"

            current_status = get_recruiter_status(
                result["filename"]
            )

            summary_data.append(
                {
                    "Rank": rank,

                    "Candidate Name":
                        result["name"],

                    "Email":
                        result["email"],

                    "Phone":
                        result["phone"],

                    "Education":
                        result["education"],

                    "CGPA":
                        result["cgpa"],

                    "Skills":
                        ", ".join(
                            result["candidate_skills"]
                        ),

                    "Final Score (%)":
                        round(
                            result["final_score"],
                            2
                        ),

                    "Skill Match (%)":
                        round(
                            result["skill_match"],
                            2
                        ),

                    "Skill Coverage (%)":
                        round(
                            result["skill_coverage"],
                            2
                        ),

                    "Semantic Similarity (%)":
                        round(
                            result["semantic"],
                            2
                        ),

                    "Text Similarity (%)":
                        round(
                            result["similarity"],
                            2
                        ),

                    "Skills Matched":
                        skill_summary,

                    "Matched Skills":
                        ", ".join(
                            result["matched_skills"]
                        ),

                    "Missing Skills":
                        ", ".join(
                            result["missing_skills"]
                        ),

                    "AI Recommendation":
                        result["recommendation"],

                    "Recruiter Decision":
                        current_status
                }
            )

        summary_df = pd.DataFrame(
            summary_data
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # CSV DOWNLOAD
        # ====================================================

        st.subheader(
            "📥 Download Candidate Summary"
        )

        csv_data = summary_df.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            label="📥 Download CSV Report",
            data=csv_data,
            file_name="candidate_screening_summary.csv",
            mime="text/csv"
        )

        st.divider()

        # ====================================================
        # VISUAL ANALYTICS
        # ====================================================

        st.header(
            "📈 Visual Candidate Analytics"
        )

        chart_data = pd.DataFrame(
            {
                "Candidate": [
                    result["name"]
                    for result in ranked_results
                ],

                "Final Score": [
                    result["final_score"]
                    for result in ranked_results
                ],

                "Skill Match": [
                    result["skill_match"]
                    for result in ranked_results
                ],

                "Skill Coverage": [
                    result["skill_coverage"]
                    for result in ranked_results
                ],

                "Semantic Similarity": [
                    result["semantic"]
                    for result in ranked_results
                ],

                "Text Similarity": [
                    result["similarity"]
                    for result in ranked_results
                ]
            }
        )

        # ----------------------------------------------------
        # FINAL SCORE
        # ----------------------------------------------------

        st.subheader(
            "🎯 Final Score Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[["Final Score"]]
        )

        # ----------------------------------------------------
        # SKILL MATCH
        # ----------------------------------------------------

        st.subheader(
            "🛠️ Skill Match Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[["Skill Match"]]
        )

        # ----------------------------------------------------
        # SKILL COVERAGE
        # ----------------------------------------------------

        st.subheader(
            "📊 Skill Coverage Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[["Skill Coverage"]]
        )

        # ----------------------------------------------------
        # SEMANTIC
        # ----------------------------------------------------

        st.subheader(
            "🧠 Semantic Similarity Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[["Semantic Similarity"]]
        )

        # ----------------------------------------------------
        # TEXT
        # ----------------------------------------------------

        st.subheader(
            "🤖 Text Similarity Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[["Text Similarity"]]
        )

        # ----------------------------------------------------
        # OVERALL
        # ----------------------------------------------------

        st.subheader(
            "🏆 Overall Candidate Comparison"
        )

        st.bar_chart(
            chart_data.set_index(
                "Candidate"
            )[
                [
                    "Final Score",
                    "Semantic Similarity",
                    "Skill Match",
                    "Skill Coverage",
                    "Text Similarity"
                ]
            ]
        )

        st.divider()

        # ====================================================
        # RECRUITER DECISION CENTER
        # ====================================================

        st.header(
            "🧑‍💼 Recruiter Decision Center"
        )

        st.write(
            "The AI recommendation is advisory. "
            "The recruiter makes the final hiring decision."
        )

        for result in ranked_results:

            filename = result["filename"]

            current_status = get_recruiter_status(
                filename
            )

            st.subheader(
                f"👤 {result['name']}"
            )

            st.write(
                f"🎯 Final Score: "
                f"**{result['final_score']:.2f}%**"
            )

            st.write(
                f"📊 Skill Coverage: "
                f"**{result['skill_coverage']:.2f}%**"
            )

            st.write(
                f"🤖 AI Recommendation: "
                f"**{result['recommendation']}**"
            )

            st.write(
                f"🧑‍💼 Current Recruiter Decision: "
                f"**{status_display(current_status)}**"
            )

            decision_options = [
                "Shortlisted",
                "Under Review",
                "Rejected"
            ]

            current_index = decision_options.index(
                current_status
            )

            selected_status = st.selectbox(
                "🧑‍💼 Recruiter Decision",
                decision_options,
                index=current_index,
                format_func=status_display,
                key=f"decision_{filename}"
            )

            if selected_status != current_status:

                set_recruiter_status(
                    filename,
                    selected_status
                )

                st.rerun()

            if selected_status == "Shortlisted":

                st.success(
                    f"🟢 {result['name']} "
                    f"has been shortlisted."
                )

            elif selected_status == "Rejected":

                st.error(
                    f"🔴 {result['name']} "
                    f"has been rejected."
                )

            else:

                st.info(
                    f"🟡 {result['name']} "
                    f"is under recruiter review."
                )

            st.divider()

        # ====================================================
        # UPDATED PIPELINE
        # ====================================================

        st.header(
            "📌 Updated Recruitment Pipeline"
        )

        shortlisted_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Shortlisted"
        )

        review_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Under Review"
        )

        rejected_count = sum(
            1
            for result in ranked_results
            if get_recruiter_status(
                result["filename"]
            ) == "Rejected"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "🟢 Shortlisted",
                shortlisted_count
            )

        with col2:

            st.metric(
                "🟡 Under Review",
                review_count
            )

        with col3:

            st.metric(
                "🔴 Rejected",
                rejected_count
            )

        st.divider()

        # ====================================================
        # SEARCH AND FILTER
        # ====================================================

        st.header(
            "🔎 Candidate Search & Filter"
        )

        st.write(
            "Find candidates using name, recruiter status, "
            "score and skills."
        )

        search_name = st.text_input(
            "🔎 Search Candidate by Name",
            placeholder="Enter candidate name..."
        )

        status_filter = st.selectbox(
            "📌 Filter by Recruiter Status",
            [
                "All",
                "Shortlisted",
                "Under Review",
                "Rejected"
            ],
            key="status_filter"
        )

        minimum_score = st.slider(
            "🎯 Minimum Final Match Score",
            min_value=0.0,
            max_value=100.0,
            value=0.0,
            step=0.01,
            format="%.2f"
        )

        all_skills = set()

        for result in ranked_results:

            for skill in result["candidate_skills"]:

                all_skills.add(
                    skill.lower()
                )

        skill_options = (
            ["All"]
            + sorted(all_skills)
        )

        selected_skill = st.selectbox(
            "🛠️ Filter by Skill",
            skill_options,
            key="skill_filter"
        )

        # ====================================================
        # APPLY FILTERS
        # ====================================================

        filtered_results = []

        for result in ranked_results:

            current_status = get_recruiter_status(
                result["filename"]
            )

            if search_name.strip():

                if (
                    search_name.lower()
                    not in result["name"].lower()
                ):

                    continue

            if status_filter != "All":

                if current_status != status_filter:

                    continue

            if result["final_score"] < minimum_score:

                continue

            if selected_skill != "All":

                candidate_skill_names = [
                    skill.lower()
                    for skill in result["candidate_skills"]
                ]

                if (
                    selected_skill.lower()
                    not in candidate_skill_names
                ):

                    continue

            filtered_results.append(result)

        # ====================================================
        # FILTER RESULTS
        # ====================================================

        st.subheader(
            "📊 Filter Results"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "👥 Matching Candidates",
                len(filtered_results)
            )

        with col2:

            st.metric(
                "📋 Total Candidates",
                len(ranked_results)
            )

        with col3:

            st.metric(
                "🎯 Minimum Score",
                f"{minimum_score:.2f}%"
            )

        if filtered_results:

            for result in filtered_results:

                current_status = get_recruiter_status(
                    result["filename"]
                )

                st.success(
                    f"👤 **{result['name']}** | "
                    f"Final Score: "
                    f"**{result['final_score']:.2f}%** | "
                    f"AI: **{result['recommendation']}** | "
                    f"Recruiter: "
                    f"**{status_display(current_status)}**"
                )

                st.write(
                    f"🛠️ Skill Match: "
                    f"{result['skill_match']:.2f}%"
                )

                st.write(
                    f"📊 Skill Coverage: "
                    f"{result['skill_coverage']:.2f}%"
                )

                st.write(
                    f"🧠 Semantic Similarity: "
                    f"{result['semantic']:.2f}%"
                )

                st.write(
                    f"🤖 Text Similarity: "
                    f"{result['similarity']:.2f}%"
                )

                st.divider()

        else:

            st.info(
                "No candidates match the selected filters."
            )

        # ====================================================
        # MISSING SKILLS ANALYSIS
        # ====================================================

        st.header(
            "🛠️ Missing Skills Analysis"
        )

        if missing_skill_count:

            sorted_missing_skills = sorted(
                missing_skill_count.items(),
                key=lambda x: x[1],
                reverse=True
            )

            for skill, count in sorted_missing_skills:

                st.write(
                    f"**{skill}** — "
                    f"missing in {count} candidate(s)"
                )

        else:

            st.success(
                "🎉 No missing skills detected!"
            )

        st.divider()

        # ====================================================
        # CANDIDATE DETAILS
        # ====================================================

        st.header(
            "👤 Candidate Details"
        )

        candidate_names = [
            result["name"]
            for result in ranked_results
        ]

        selected_candidate_name = st.selectbox(
            "🔽 Select Candidate",
            candidate_names,
            key="candidate_selector"
        )

        selected_candidate = None

        for result in ranked_results:

            if result["name"] == selected_candidate_name:

                selected_candidate = result

                break

        if selected_candidate is not None:

            st.subheader(
                f"👤 {selected_candidate['name']}"
            )

            col1, col2 = st.columns(2)

            with col1:

                st.write(
                    f"📧 **Email:** "
                    f"{selected_candidate['email']}"
                )

                st.write(
                    f"📞 **Phone:** "
                    f"{selected_candidate['phone']}"
                )

                st.write(
                    f"🎓 **Education:** "
                    f"{selected_candidate['education']}"
                )

            with col2:

                st.write(
                    f"📈 **CGPA:** "
                    f"{selected_candidate['cgpa']}"
                )

                selected_skills = (
                    ", ".join(
                        selected_candidate[
                            "candidate_skills"
                        ]
                    )
                    if selected_candidate[
                        "candidate_skills"
                    ]
                    else "Not found"
                )

                st.write(
                    f"🛠️ **Skills:** "
                    f"{selected_skills}"
                )

            st.divider()

            # ------------------------------------------------
            # SCORE BREAKDOWN
            # ------------------------------------------------

            st.subheader(
                "📊 Screening Score Breakdown"
            )

            col1, col2, col3, col4 = st.columns(4)

            with col1:

                st.metric(
                    "🎯 Final Score",
                    f"{selected_candidate['final_score']:.2f}%"
                )

            with col2:

                st.metric(
                    "🛠️ Skill Match",
                    f"{selected_candidate['skill_match']:.2f}%"
                )

            with col3:

                st.metric(
                    "📊 Skill Coverage",
                    f"{selected_candidate['skill_coverage']:.2f}%"
                )

            with col4:

                st.metric(
                    "🧠 Semantic Similarity",
                    f"{selected_candidate['semantic']:.2f}%"
                )

            st.metric(
                "🤖 Text Similarity",
                f"{selected_candidate['similarity']:.2f}%"
            )

            st.progress(
                min(
                    max(
                        selected_candidate["final_score"] / 100,
                        0.0
                    ),
                    1.0
                )
            )

            # ------------------------------------------------
            # RECRUITER DECISION
            # ------------------------------------------------

            st.subheader(
                "🧑‍💼 Recruiter Decision"
            )

            selected_status = get_recruiter_status(
                selected_candidate["filename"]
            )

            st.write(
                f"Current Decision: "
                f"**{status_display(selected_status)}**"
            )

            # ------------------------------------------------
            # MATCHED SKILLS
            # ------------------------------------------------

            st.subheader(
                "✅ Matched Skills"
            )

            if selected_candidate["matched_skills"]:

                for skill in selected_candidate[
                    "matched_skills"
                ]:

                    st.success(
                        skill.upper()
                    )

            else:

                st.info(
                    "No matched skills."
                )

            # ------------------------------------------------
            # MISSING SKILLS
            # ------------------------------------------------

            st.subheader(
                "❌ Missing Skills"
            )

            if selected_candidate["missing_skills"]:

                for skill in selected_candidate[
                    "missing_skills"
                ]:

                    st.error(
                        skill.upper()
                    )

            else:

                st.success(
                    "🎉 No missing skills!"
                )

            # ------------------------------------------------
            # SKILL COVERAGE
            # ------------------------------------------------

            st.subheader(
                "📊 Skill Coverage Explanation"
            )

            st.info(
                generate_skill_coverage_explanation(
                    selected_candidate
                )
            )

            # ------------------------------------------------
            # AI RECOMMENDATION
            # ------------------------------------------------

            st.subheader(
                "📌 AI Recommendation"
            )

            recommendation = selected_candidate[
                "recommendation"
            ]

            if "Excellent" in recommendation:

                st.success(
                    "🟢 Excellent Match"
                )

            elif "Moderate" in recommendation:

                st.warning(
                    "🟡 Moderate Match"
                )

            else:

                st.error(
                    "🔴 Low Match"
                )

            # ------------------------------------------------
            # AI EXPLANATION
            # ------------------------------------------------

            st.subheader(
                "🤖 AI Screening Explanation"
            )

            explanation = generate_ai_explanation(
                selected_candidate
            )

            for item in explanation:

                st.write(item)

        st.divider()

        # ====================================================
        # RECOMMENDED CANDIDATE
        # ====================================================

        st.header(
            "🏆 Recommended Candidate"
        )

        best_status = get_recruiter_status(
            best_candidate["filename"]
        )

        st.success(
            f"**{best_candidate['name']}** is currently "
            f"the highest-ranked candidate with a final "
            f"match score of "
            f"**{best_candidate['final_score']:.2f}%**."
        )

        st.write(
            f"🛠️ Skill Coverage: "
            f"**{best_candidate['skill_coverage']:.2f}%**"
        )

        st.write(
            f"🤖 AI Recommendation: "
            f"**{best_candidate['recommendation']}**"
        )

        st.write(
            f"🧑‍💼 Recruiter Decision: "
            f"**{status_display(best_status)}**"
        )

        st.divider()

        # ====================================================
        # PDF REPORT
        # ====================================================

        st.header(
            "📄 Recruiter Report"
        )

        st.write(
            "Generate a complete PDF report containing "
            "candidate rankings, screening scores, skills, "
            "AI explanations and recruiter decisions."
        )

        try:

            pdf_report = generate_pdf_report(
                ranked_results=ranked_results,
                total_candidates=total_candidates,
                average_score=average_score,
                best_candidate=best_candidate,
                shortlisted_count=shortlisted_count,
                review_count=review_count,
                rejected_count=rejected_count
            )

            st.download_button(
                label="📥 Download Recruiter Report (PDF)",
                data=pdf_report,
                file_name="AI_Recruiter_Screening_Report.pdf",
                mime="application/pdf"
            )

        except Exception as error:

            st.error(
                f"Could not generate PDF report: {error}"
            )

    # ========================================================
    # RANK CANDIDATES & CONFIGURE AGENT CONTEXT
    # ========================================================

    if results:
        # Attach latest recruiter decisions
        for candidate in results:
            candidate["recruiter_decision"] = get_recruiter_status(
                candidate["filename"]
            )

        # Rank candidates by final_score descending
        ranked_results = sorted(
            results,
            key=lambda x: x["final_score"],
            reverse=True
        )

        # Synchronize context with Recruitment Agent
        recruitment_agent.set_context(
            job_description=jd_text,
            candidates=ranked_results
        )

        # ====================================================
        # TOP-LEVEL RESULTS TABS (NO FULL SCROLLING NEEDED)
        # ====================================================
        st.markdown("---")
        tab_dashboard, tab_agent, tab_profiles = st.tabs([
            "📊 Recruiter Dashboard & Analytics",
            "🤖 AI Recruitment Agent Assistant",
            "👤 Candidate Profiles & Resumes"
        ])

        # ----------------------------------------------------
        # TAB 1: RECRUITER DASHBOARD & ANALYTICS
        # ----------------------------------------------------
        with tab_dashboard:
            render_recruiter_dashboard(ranked_results)

        # ----------------------------------------------------
        # TAB 2: AI RECRUITMENT AGENT ASSISTANT
        # ----------------------------------------------------
        with tab_agent:
            st.header("🤖 AI Recruitment Agent Assistant")
            st.markdown(
                "Interact directly with your autonomous AI screening agent. "
                "Ask for candidate rankings, head-to-head comparisons, hiring recommendations, "
                "interview questions, or deep-dive into skill gaps."
            )

            # Executive Insights & Quick Stats
            summary_tool = recruitment_agent.available_tools.get("generate_recruitment_summary")
            summary_stats = summary_tool(ranked_results) if summary_tool else {}

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Total Screened", summary_stats.get("total_candidates", len(ranked_results)))
            with col_s2:
                st.metric("Average Score", f"{summary_stats.get('average_score', 0):.2f}%")
            with col_s3:
                top_cand = summary_stats.get("top_candidate")
                if isinstance(top_cand, dict):
                    top_name = top_cand.get("name", "N/A") 
                    top_score = top_cand.get("final_score", 0)
                else:
                    top_name = str(top_cand) if top_cand else "N/A"
                    top_score = 0
                st.metric("Top Candidate", top_name, f"{top_score:.2f}%")
            with col_s4:
                st.metric("Shortlisted", summary_stats.get("shortlisted_count", 0))

            st.divider()

            # Quick Action Prompt Buttons
            st.subheader("⚡ Quick Agent Prompts")
            qcol1, qcol2, qcol3, qcol4 = st.columns(4)
            agent_prompt_action = None

            with qcol1:
                if st.button("🏆 Best Candidate?", use_container_width=True, key="btn_agent_best"):
                    agent_prompt_action = "Who is the best candidate and why should we interview them?"

            with qcol2:
                if st.button("⚖️ Compare Top 2", use_container_width=True, key="btn_agent_compare"):
                    agent_prompt_action = "Compare the top 2 candidates side-by-side with pros and cons."

            with qcol3:
                if st.button("🎯 Shortlist Actions", use_container_width=True, key="btn_agent_shortlist"):
                    agent_prompt_action = "Give me a summary of shortlisted candidates and recommended next steps."

            with qcol4:
                if st.button("❓ Interview Questions", use_container_width=True, key="btn_agent_questions"):
                    agent_prompt_action = "What technical interview questions should we ask the top candidate?"

            # If user clicked a quick prompt
            if agent_prompt_action:
                st.session_state.agent_chat_history.append({"role": "user", "content": agent_prompt_action})
                with st.spinner("🤖 Agent is analyzing..."):
                    agent_res = recruitment_agent.handle_query(
                        agent_prompt_action,
                    )
                    st.session_state.agent_chat_history.append({"role": "assistant", "content": agent_res["response"]})
                st.rerun()

            st.divider()
            st.subheader("💬 Recruiter Agent Chat")

            # Display Chat History
            for message in st.session_state.agent_chat_history:
                msg_role = "user" if message["role"] == "user" else "assistant"
                with st.chat_message(msg_role):
                    st.markdown(message["content"])

            # Chat Input Field
            chat_input = st.chat_input(
                "Ask the agent (e.g. 'Compare Alice and Bob', 'Who knows Spring Boot?', 'Why was Candidate X rejected?')...",
                key="recruiter_agent_chat_input"
            )
            if chat_input:
                st.session_state.agent_chat_history.append({"role": "user", "content": chat_input})
                with st.chat_message("user"):
                    st.markdown(chat_input)
                with st.chat_message("assistant"):
                    with st.spinner("🤖 Agent is reasoning..."):
                        agent_res = recruitment_agent.handle_query(
                            chat_input,
                            api_key=gemini_api_key if gemini_api_key else None
                        )
                        st.markdown(agent_res["response"])
                st.session_state.agent_chat_history.append({"role": "assistant", "content": agent_res["response"]})
                st.rerun()

        # ----------------------------------------------------
        # TAB 3: CANDIDATE PROFILES & RESUMES
        # ----------------------------------------------------
        with tab_profiles:
            st.header("👤 Candidate Profiles & Resumes")
            st.markdown("Detailed breakdown of extracted credentials, skills, and resumes for every screened candidate.")

            for rank, result in enumerate(ranked_results, start=1):
                rec_class = (
                    "shortlisted-card"
                    if "Shortlisted" in str(result.get("recommendation", ""))
                    else "review-card"
                    if "Review" in str(result.get("recommendation", ""))
                    else "rejected-card"
                )

                st.markdown(
                    f"""
                    <div class="candidate-card {rec_class}">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <h3 style="margin:0;">#{rank} - {result.get('name', 'Unknown')}</h3>
                            <span class="recommendation-badge">{result.get('recommendation', '')}</span>
                        </div>
                        <p style="margin:5px 0 0 0; color:#cbd5e1;">📄 <b>File:</b> {result.get('filename', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
                with metric_col1:
                    st.metric("Final Match Score", f"{result.get('final_score', 0.0):.2f}%")
                with metric_col2:
                    st.metric("Skill Match (50%)", f"{result.get('skill_match', 0.0):.2f}%")
                with metric_col3:
                    st.metric("Semantic Match (30%)", f"{result.get('semantic', 0.0):.2f}%")
                with metric_col4:
                    st.metric("Skill Coverage", f"{result.get('skill_coverage', 0.0):.2f}%")

                details_col1, details_col2 = st.columns(2)
                with details_col1:
                    st.markdown(f"**📧 Email:** {result.get('email', 'Not found')}")
                    st.markdown(f"**📞 Phone:** {result.get('phone', 'Not found')}")
                    st.markdown(f"**🎓 Education:** {result.get('education', 'Not found')}")
                with details_col2:
                    st.markdown(f"**💼 Experience:** {result.get('experience', 'Not found')}")
                    st.markdown(f"**📈 CGPA:** {result.get('cgpa', 'Not found')}")
                    st.markdown(f"**🛠️ Total Skills Extracted:** {len(result.get('candidate_skills', []))}")

                with st.expander(f"🔍 Skill Breakdown for {result.get('name', 'Candidate')}"):
                    matched = result.get('matched_skills', [])
                    missing = result.get('missing_skills', [])
                    st.markdown(f"**Matched Skills ({len(matched)}):** {', '.join(matched) if matched else 'None'}")
                    st.markdown(f"**Missing Skills ({len(missing)}):** {', '.join(missing) if missing else 'None'}")

                with st.expander(f"📄 View Extracted Text for {result.get('name', 'Candidate')}"):
                    st.text_area("Cleaned Text", result.get("cleaned_text", ""), height=150, key=f"cleaned_{result.get('filename', '')}_{rank}")
                    st.text_area("Raw Text", result.get("raw_text", ""), height=150, key=f"raw_{result.get('filename', '')}_{rank}")

                st.markdown("---")


# ============================================================
# NO FILES / VALIDATION MESSAGES
# ============================================================

elif (
    uploaded_jd is None
    and not uploaded_resumes
):

    st.info(
        "👆 Upload a Job Description and candidate "
        "resumes to start screening."
    )

elif uploaded_jd is None:

    st.warning(
        "Please upload a Job Description first."
    )

elif not uploaded_resumes:

    st.warning(
        "Please upload at least one candidate resume."
    )

elif jd_text is not None and not jd_text.strip():

    st.error(
        "The uploaded Job Description is empty. "
        "Please upload a valid Job Description."
    )