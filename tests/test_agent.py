"""
Unit and Integration Test for AI Recruitment Agent and Tool Layer
"""

import sys
import os

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.agent_tools import (
    AGENT_TOOLS,
    TOOL_SCHEMAS,
    screen_candidate,
    rank_candidates,
    compare_candidates,
    find_candidates_by_skill,
    explain_candidate,
    recommend_hiring_actions,
    generate_recruitment_summary
)
from agent.recruitment_agent import recruitment_agent


def test_agent_tools_and_reasoning():
    print("Testing Agent Tools & Reasoning...")

    jd = """
    Senior Java Developer required with Java, Spring Boot, REST API, SQL, and Git.
    Experience with Microservices and Docker is a plus.
    """

    resume1 = """
    Rahul Sharma
    Email: rahul.sharma@example.com
    Phone: +91 9876543210
    Education: B.Tech in Computer Science, CGPA: 8.9
    Skills: Java, Spring Boot, REST API, MySQL, Git, Docker
    Experience: 3 years developing enterprise backend services with Java and Spring Boot.
    """

    resume2 = """
    Priya Patel
    Email: priya.patel@example.com
    Phone: +91 9123456789
    Education: B.E in Information Technology, CGPA: 7.5
    Skills: Python, Machine Learning, TensorFlow, SQL
    Experience: 2 years building machine learning data pipelines.
    """

    # 1. Test screen_candidate
    res1 = screen_candidate(resume1, jd)
    res2 = screen_candidate(resume2, jd)

    assert res1["final_score"] > 0, "Rahul should have positive final score"
    assert res2["final_score"] > 0, "Priya should have positive final score"
    assert res1["final_score"] > res2["final_score"], "Rahul should score higher than Priya for Java JD"
    print(f"✓ screen_candidate passed (Rahul: {res1['final_score']}%, Priya: {res2['final_score']}%)")

    # 2. Test rank_candidates
    candidates = [res2, res1]
    ranked = rank_candidates(candidates)
    assert ranked[0]["name"] == "Rahul Sharma", "Rahul should be ranked #1"
    print(f"✓ rank_candidates passed (#1: {ranked[0]['name']})")

    # 3. Test compare_candidates
    comp = compare_candidates(res1, res2)
    assert comp["winner"] == "Rahul Sharma"
    print(f"✓ compare_candidates passed (Winner: {comp['winner']}, Verdict: {comp['verdict']})")

    # 4. Test find_candidates_by_skill
    found = find_candidates_by_skill(ranked, "python")
    assert any(c["name"] == "Priya Patel" for c in found), "Priya should be found for Python"
    print(f"✓ find_candidates_by_skill passed (Found {len(found)} candidate for Python)")

    # 5. Test explain_candidate
    explanation = explain_candidate(res1)
    assert len(explanation) > 0
    print(f"✓ explain_candidate passed ({len(explanation)} points)")

    # 6. Test recommend_hiring_actions
    actions = recommend_hiring_actions(res1, jd)
    assert "recommended_action" in actions
    print(f"✓ recommend_hiring_actions passed (Action: {actions['recommended_action']})")

    # 7. Test generate_recruitment_summary
    summary = generate_recruitment_summary(ranked)
    assert summary["total_candidates"] == 2
    print(f"✓ generate_recruitment_summary passed (Total: {summary['total_candidates']})")

    # 8. Test RecruitmentAgent query handling
    recruitment_agent.set_context(job_description=jd, candidates=ranked)
    
    # Query: rank
    ans_rank = recruitment_agent.handle_query("Who are the top candidates?")
    assert "Leaderboard" in ans_rank["response"]
    print("✓ Agent query 'rank' passed")

    # Query: compare
    ans_comp = recruitment_agent.handle_query("Compare Rahul and Priya")
    assert "Head-to-Head" in ans_comp["response"]
    print("✓ Agent query 'compare' passed")

    # Query: interview questions
    ans_q = recruitment_agent.handle_query("What interview questions should I ask Rahul?")
    assert "Hiring Actions" in ans_q["response"]
    print("✓ Agent query 'interview questions' passed")

    # Query: summary
    ans_sum = recruitment_agent.handle_query("Give me a recruitment summary")
    assert "Executive Summary" in ans_sum["response"]
    print("✓ Agent query 'summary' passed")

    print("\nALL AGENT & TOOL TESTS PASSED SUCCESSFULLY!")


if __name__ == "__main__":
    test_agent_tools_and_reasoning()
