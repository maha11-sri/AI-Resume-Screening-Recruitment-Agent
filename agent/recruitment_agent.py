"""
AI Recruitment Agent - Core Orchestration and Reasoning Engine

This module defines the RecruitmentAgent class, providing an autonomous agent
that manages candidate screening, maintains conversational context, dispatches
tools, and answers complex recruiter queries with either its built-in local
reasoning engine or an optional external LLM (e.g. Google Gemini / OpenAI).
"""

import json
import re
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from agent.agent_tools import (
    parse_resume,
    clean_resume_text,
    extract_candidate_information,
    extract_job_skills,
    analyze_candidate_skills,
    calculate_text_similarity,
    calculate_semantic_match,
    screen_candidate,
    rank_candidates,
    find_candidates_by_skill,
    explain_candidate,
    compare_candidates,
    filter_candidates,
    recommend_hiring_actions,
    generate_recruitment_summary,
    AGENT_TOOLS,
    TOOL_SCHEMAS
)


class RecruitmentAgent:
    """
    Autonomous AI Recruitment Agent for resume screening, candidate ranking,
    head-to-head comparison, and conversational recruiter assistance.
    """

    def __init__(self, name: str = "AI Recruitment Agent"):
        self.name = name
        self.job_description: str = ""
        self.candidates: List[Dict[str, Any]] = []
        self.conversation_history: List[Dict[str, str]] = []
        self.available_tools = AGENT_TOOLS
        self.tool_schemas = TOOL_SCHEMAS

    # ========================================================
    # 1. CONTEXT MANAGEMENT
    # ========================================================

    def set_context(
        self,
        job_description: Optional[str] = None,
        candidates: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """Update the agent's working memory with current JD and candidate data."""
        if job_description is not None:
            self.job_description = job_description
        if candidates is not None:
            self.candidates = candidates

    def clear_memory(self) -> None:
        """Reset conversation history."""
        self.conversation_history = []

    # ========================================================
    # 2. SCREENING PIPELINE
    # ========================================================

    def screen_one(
        self,
        resume_text: str,
        job_description: Optional[str] = None
    ) -> Dict[str, Any]:
        """Screen a single candidate resume against a job description."""
        jd = job_description or self.job_description
        if not jd:
            raise ValueError("No job description available for screening.")
        return screen_candidate(resume_text, jd)

    def run_screening_pipeline(
        self,
        job_description: str,
        uploaded_resumes: List[Any],
        recruiter_decisions: Optional[Dict[str, str]] = None
    ) -> List[Dict[str, Any]]:
        """
        Orchestrates full batch screening across all uploaded resume files.
        Extracts, cleans, screens, links recruiter decisions, and ranks candidates.
        """
        self.set_context(job_description=job_description)
        decisions = recruiter_decisions or {}
        screened_list = []

        for resume_file in uploaded_resumes:
            filename = getattr(resume_file, "name", "unknown_resume")
            try:
                # 1. Parse document
                raw_text = parse_resume(resume_file)
                # 2. Clean text
                cleaned = clean_resume_text(raw_text)
                # 3. Screen candidate
                result = screen_candidate(raw_text, job_description)
                # 4. Associate filename and recruiter decision
                result["filename"] = filename
                result["raw_text"] = raw_text
                result["cleaned_text"] = cleaned
                result["recruiter_decision"] = decisions.get(filename, "Under Review")
                screened_list.append(result)
            except Exception as error:
                screened_list.append({
                    "filename": filename,
                    "name": "Parse Error",
                    "email": "N/A",
                    "phone": "N/A",
                    "education": "N/A",
                    "cgpa": "N/A",
                    "candidate_skills": [],
                    "matched_skills": [],
                    "missing_skills": [],
                    "skill_match": 0.0,
                    "skill_coverage": 0.0,
                    "similarity": 0.0,
                    "semantic": 0.0,
                    "final_score": 0.0,
                    "recommendation": "🔴 Low Match",
                    "recruiter_decision": decisions.get(filename, "Under Review"),
                    "error": str(error)
                })

        # Rank all screened results
        ranked = rank_candidates(screened_list)
        self.set_context(candidates=ranked)
        return ranked

    # ========================================================
    # 3. CONVERSATIONAL REASONING & INTENT DISPATCHER
    # ========================================================

    def handle_query(
        self,
        query: str,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Main entry point for conversational agent requests.
        Decides the intent, invokes appropriate tools, and synthesizes a response.
        If an API key is provided, attempts LLM synthesis; otherwise runs local reasoning.
        """
        query_text = (query or "").strip()
        if not query_text:
            return {
                "response": "Hello! I am your AI Recruitment Agent. Upload a Job Description and resumes, or ask me to rank, compare, or explain candidates.",
                "tool_used": None,
                "data": None
            }

        # Step 1: Detect intent and extract parameters
        intent_info = self._analyze_query_intent(query_text)
        action = intent_info["action"]
        params = intent_info["params"]

        # Step 2: Dispatch and execute tool
        tool_name, tool_result = self._execute_tool(action, params, query_text)

        # Step 3: Synthesize human-readable response
        if api_key and api_key.strip():
            llm_response = self._call_gemini_llm(query_text, tool_name, tool_result, api_key.strip())
            if llm_response:
                final_text = llm_response
            else:
                final_text = self._synthesize_local_response(action, tool_result, query_text, params)
        else:
            final_text = self._synthesize_local_response(action, tool_result, query_text, params)

        # Step 4: Record memory
        self.conversation_history.append({"role": "user", "content": query_text})
        self.conversation_history.append({"role": "agent", "content": final_text, "tool_used": tool_name})

        return {
            "response": final_text,
            "tool_used": tool_name,
            "data": tool_result
        }

    # ========================================================
    # 4. INTENT & ENTITY EXTRACTION
    # ========================================================

    def _analyze_query_intent(self, query: str) -> Dict[str, Any]:
        """Classify user query into actionable agent tasks and extract entities."""
        q = query.lower()

        # Check for comparison
        if any(w in q for w in ["compare", "versus", "vs", "difference between", "better candidate"]):
            cand_names = self._find_candidate_names_in_text(query)
            return {"action": "compare", "params": {"candidates": cand_names}}

        # Check for interview recommendations / hiring actions
        if any(w in q for w in ["question", "interview", "hire", "hiring action", "action", "next step", "probe"]):
            cand_names = self._find_candidate_names_in_text(query)
            return {"action": "recommend_actions", "params": {"candidate": cand_names[0] if cand_names else None}}

        # Check for explanation / reason / diagnosis
        if any(w in q for w in ["why", "explain", "reason", "diagnos", "justif", "shortlist", "reject"]):
            cand_names = self._find_candidate_names_in_text(query)
            return {"action": "explain", "params": {"candidate": cand_names[0] if cand_names else None}}

        # Check for ranking / top candidates
        if any(w in q for w in ["top", "best", "rank", "leaderboard", "highest", "first", "winner"]):
            # Check for count like "top 3", "top 5"
            count_match = re.search(r"top\s+(\d+)", q)
            count = int(count_match.group(1)) if count_match else None
            return {"action": "rank", "params": {"limit": count}}

        # Check for skill lookup or missing skills
        if any(w in q for w in ["missing skill", "lacking", "skill gap"]):
            return {"action": "missing_skills", "params": {}}

        # Check for specific skill filtering
        skills_found = self._find_skills_in_text(q)
        if skills_found or any(w in q for w in ["skill", "knows", "has", "who has", "filter"]):
            return {"action": "find_by_skill", "params": {"skills": skills_found}}

        # Check for summary / pipeline status
        if any(w in q for w in ["summary", "overview", "status", "stats", "pipeline", "report"]):
            return {"action": "summary", "params": {}}

        # Default fallback
        return {"action": "general_inquiry", "params": {}}

    def _find_candidate_names_in_text(self, text: str) -> List[str]:
        """Find candidate names mentioned in query based on currently loaded candidates."""
        found = []
        text_lower = text.lower()
        for cand in self.candidates:
            name = cand.get("name", "").strip()
            if not name or name == "Not found" or name == "Unknown":
                continue
            # Match first name or full name
            parts = name.split()
            first_name = parts[0].lower() if parts else ""
            if name.lower() in text_lower or (len(first_name) >= 3 and first_name in text_lower):
                found.append(name)
        return found

    def _find_skills_in_text(self, text: str) -> List[str]:
        """Identify technical skills mentioned in the user query."""
        from models.skill_matcher import COMMON_SKILLS, contains_skill
        matches = []
        for skill in COMMON_SKILLS:
            if contains_skill(text, skill):
                matches.append(skill)
        return matches

    # ========================================================
    # 5. TOOL EXECUTION LAYER
    # ========================================================

    def _execute_tool(
        self,
        action: str,
        params: Dict[str, Any],
        query: str
    ) -> tuple[str, Any]:
        """Dispatches parameters to registered agent tools."""
        if not self.candidates:
            return "no_candidates", "No candidate resumes have been screened yet. Please upload a job description and candidate resumes first."

        if action == "rank":
            ranked = rank_candidates(self.candidates)
            limit = params.get("limit")
            if limit and limit > 0:
                ranked = ranked[:limit]
            return "rank_candidates", ranked

        elif action == "compare":
            names = params.get("candidates", [])
            cand_a, cand_b = None, None
            if len(names) >= 2:
                cand_a = next((c for c in self.candidates if c.get("name") == names[0]), None)
                cand_b = next((c for c in self.candidates if c.get("name") == names[1]), None)

            if not cand_a or not cand_b:
                # Default to top 2 candidates
                ranked = rank_candidates(self.candidates)
                if len(ranked) >= 2:
                    cand_a, cand_b = ranked[0], ranked[1]
                elif len(ranked) == 1:
                    cand_a = ranked[0]
                    return "compare_candidates", {"error": "Only one candidate is uploaded; comparison requires at least two."}
                else:
                    return "compare_candidates", {"error": "No candidates available to compare."}

            comparison = compare_candidates(cand_a, cand_b)
            return "compare_candidates", comparison

        elif action == "explain":
            cand_name = params.get("candidate")
            target = None
            if cand_name:
                target = next((c for c in self.candidates if cand_name.lower() in c.get("name", "").lower()), None)
            if not target and self.candidates:
                target = rank_candidates(self.candidates)[0]

            if not target:
                return "explain_candidate", "No candidate found to explain."

            explanation_lines = explain_candidate(target)
            return "explain_candidate", {"candidate": target.get("name"), "details": explanation_lines, "candidate_obj": target}

        elif action == "recommend_actions":
            cand_name = params.get("candidate")
            target = None
            if cand_name:
                target = next((c for c in self.candidates if cand_name.lower() in c.get("name", "").lower()), None)
            if not target and self.candidates:
                target = rank_candidates(self.candidates)[0]

            if not target:
                return "recommend_hiring_actions", "No candidate found for recommendations."

            rec = recommend_hiring_actions(target, self.job_description)
            return "recommend_hiring_actions", rec

        elif action == "find_by_skill":
            skills = params.get("skills", [])
            if skills:
                primary_skill = skills[0]
                matched = find_candidates_by_skill(self.candidates, primary_skill)
                return "find_candidates_by_skill", {"skill": primary_skill, "results": matched}
            else:
                return "filter_candidates", self.candidates

        elif action == "missing_skills":
            summary = generate_recruitment_summary(self.candidates)
            return "generate_recruitment_summary", summary

        elif action == "summary":
            summary = generate_recruitment_summary(self.candidates)
            return "generate_recruitment_summary", summary

        else:
            # Default to recruitment summary
            summary = generate_recruitment_summary(self.candidates)
            return "generate_recruitment_summary", summary

    # ========================================================
    # 6. LOCAL RESPONSE SYNTHESIZER
    # ========================================================

    def _synthesize_local_response(
        self,
        action: str,
        tool_result: Any,
        query: str,
        params: Dict[str, Any]
    ) -> str:
        """Converts structured tool output into a rich, professional Markdown response."""
        if isinstance(tool_result, str):
            return tool_result

        if action == "rank":
            ranked = tool_result
            if not ranked:
                return "No candidates found to rank."

            medals = ["🥇", "🥈", "🥉"]
            lines = [f"### 🏆 Candidate Leaderboard ({len(ranked)} candidate{'s' if len(ranked) > 1 else ''})", ""]
            for idx, c in enumerate(ranked):
                prefix = medals[idx] if idx < 3 else f"**#{idx + 1}**"
                name = c.get("name", "Unknown")
                score = c.get("final_score", 0.0)
                rec = c.get("recommendation", "")
                coverage = c.get("skill_coverage", 0.0)
                skills = ", ".join(c.get("matched_skills", [])) or "None"
                decision = c.get("recruiter_decision", "Under Review")
                lines.append(f"{prefix} **{name}** — Final Score: **{score:.2f}%** ({rec})")
                lines.append(f"  - 🛠️ **Matched Skills**: {skills}")
                lines.append(f"  - 📊 **Skill Coverage**: {coverage:.2f}% | 🧑‍💼 **Decision**: {decision}")
                lines.append("")

            top = ranked[0]
            lines.append(f"💡 **Agent Recommendation**: **{top.get('name')}** is currently your top match with a score of {top.get('final_score'):.2f}%.")
            return "\n".join(lines)

        elif action == "compare":
            comp = tool_result
            if "error" in comp:
                return f"⚠️ {comp['error']}"

            cand_a = comp["candidate_a"]
            cand_b = comp["candidate_b"]
            lines = [
                f"### ⚖️ Head-to-Head Comparison: {cand_a} vs {cand_b}",
                "",
                f"| Metric | {cand_a} | {cand_b} |",
                f"| :--- | :---: | :---: |",
                f"| **Final Match Score** | **{comp['score_a']:.2f}%** | **{comp['score_b']:.2f}%** |",
                f"| **Skill Match** | {comp['skill_match_a']:.2f}% | {comp['skill_match_b']:.2f}% |",
                f"| **Skill Coverage** | {comp['skill_coverage_a']:.2f}% | {comp['skill_coverage_b']:.2f}% |",
                f"| **Semantic Similarity** | {comp['semantic_a']:.2f}% | {comp['semantic_b']:.2f}% |",
                f"| **Text Similarity** | {comp['similarity_a']:.2f}% | {comp['similarity_b']:.2f}% |",
                "",
                f"**🎯 Verdict**: {comp['verdict']}",
                ""
            ]

            if comp["unique_to_a"]:
                lines.append(f"- **{cand_a} exclusive skills**: {', '.join(comp['unique_to_a'])}")
            if comp["unique_to_b"]:
                lines.append(f"- **{cand_b} exclusive skills**: {', '.join(comp['unique_to_b'])}")
            if comp["common_skills"]:
                lines.append(f"- **Common overlapping skills**: {', '.join(comp['common_skills'])}")

            return "\n".join(lines)

        elif action == "explain":
            exp = tool_result
            name = exp.get("candidate", "Candidate")
            details = exp.get("details", [])
            lines = [f"### 🔍 Detailed Screening Diagnosis: {name}", ""]
            for d in details:
                lines.append(f"- {d}")
            return "\n".join(lines)

        elif action == "recommend_actions":
            rec = tool_result
            lines = [
                f"### 💡 Hiring Actions & Interview Plan: {rec['candidate_name']}",
                "",
                f"- **Recommended Next Step**: {rec['recommended_action']}",
                f"- **Hiring Risk Level**: **{rec['risk_level']}**",
                "",
                "#### Suggested Technical Questions:"
            ]
            for q_item in rec.get("recommended_questions", []):
                lines.append(f"1. {q_item}")

            if rec.get("missing_skill_concerns"):
                lines.append(f"\n⚠️ **Key Skill Gaps to Address**: {', '.join(rec['missing_skill_concerns'])}")

            return "\n".join(lines)

        elif action == "find_by_skill":
            res = tool_result
            skill = res.get("skill", "").title()
            matches = res.get("results", [])
            if not matches:
                return f"🔍 No candidates were found with verified skills in **{skill}**."

            lines = [f"### 🛠️ Candidates with {skill} ({len(matches)} match{'es' if len(matches) > 1 else ''})", ""]
            for c in matches:
                lines.append(f"- **{c.get('name')}** — Final Score: {c.get('final_score'):.2f}% | Recommendation: {c.get('recommendation')}")
            return "\n".join(lines)

        else:
            summary = tool_result
            lines = [
                "### 📊 Recruitment Pipeline Executive Summary",
                "",
                f"- **Total Candidates Evaluated**: {summary['total_candidates']}",
                f"- **Average Match Score**: {summary['average_score']:.2f}%",
                f"- **Top Ranked Candidate**: **{summary['top_candidate'] or 'N/A'}** ({summary['top_candidate_score']:.2f}%)",
                f"- **Match Breakdown**: 🟢 {summary['excellent_matches']} Excellent | 🟡 {summary['moderate_matches']} Moderate | 🔴 {summary['low_matches']} Low",
                ""
            ]
            if summary.get("most_common_missing_skills"):
                lines.append("#### Most Common Skill Gaps Across Applicants:")
                for skill, count in summary["most_common_missing_skills"]:
                    lines.append(f"- **{skill.title()}**: Missing in {count} applicant(s)")

            return "\n".join(lines)

    # ========================================================
    # 7. OPTIONAL GEMINI LLM REASONING
    # ========================================================

    def _call_gemini_llm(
        self,
        query: str,
        tool_name: str,
        tool_result: Any,
        api_key: str
    ) -> Optional[str]:
        """Calls Google Gemini API with the tool execution observation to generate conversational responses."""
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

        system_instruction = (
            "You are an expert, professional AI Recruitment Agent pair-programming and screening resumes for a hiring manager. "
            "You have access to candidate data and tool execution outputs. "
            "Synthesize clear, direct, insightful recruitment advice with markdown formatting. "
            "Do not fabricate candidate names or scores; only use the facts provided."
        )

        prompt_payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": f"{system_instruction}\n\n"
                                    f"Recruiter Question: {query}\n"
                                    f"Tool Executed: {tool_name}\n"
                                    f"Tool Output / Data: {json.dumps(tool_result, default=str)}\n\n"
                                    f"Please provide an intelligent, well-structured response to the recruiter."
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.3,
                "maxOutputTokens": 800
            }
        }

        try:
            req = urllib.request.Request(
                url,
                data=json.dumps(prompt_payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=10) as response:
                if response.status == 200:
                    resp_data = json.loads(response.read().decode("utf-8"))
                    candidates = resp_data.get("candidates", [])
                    if candidates:
                        parts = candidates[0].get("content", {}).get("parts", [])
                        if parts:
                            return parts[0].get("text", "").strip()
        except Exception:
            # Fall back to local synthesis on error
            return None

        return None

    # ========================================================
    # 8. STATUS & CAPABILITIES
    # ========================================================

    def get_status(self) -> Dict[str, Any]:
        """Return the live status and operational health of the agent."""
        return {
            "agent_name": self.name,
            "status": "ready",
            "active_candidates_count": len(self.candidates),
            "job_description_loaded": bool(self.job_description.strip()),
            "conversation_turns": len(self.conversation_history) // 2,
            "available_tools": list(self.available_tools.keys()),
            "tool_count": len(self.available_tools)
        }


# Singleton instance
recruitment_agent = RecruitmentAgent()