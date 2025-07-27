from google.adk.agents import Agent
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from pydantic import BaseModel
from typing import List, Literal
from teacher_assistant_agent.firestore import db

class ConceptProgress(BaseModel):
    concept: str
    initial_status: str
    post_reinforcement_status: str
    current_understanding: Literal["Improved", "Same", "Needs Attention"]

class StudentProgressReport(BaseModel):
    student_id: str
    class_name: str
    subject_name: str
    report_date: str
    chapter_name: str
    overall_progress: Literal["Excellent", "Good", "Moderate", "Needs Improvement"]
    strengths: List[str]
    persistent_weaknesses: List[str]
    concept_progress: List[ConceptProgress]
    recommendations: List[str]
    parent_summary: str

def store_progress_report(callback_context: CallbackContext):
    reports = callback_context.state.get("student_progress_report", [])
    report = callback_context.state.get("new_student_progress_report")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not report:
        print("No progress report found in state.")
        return

    doc_ref = db.collection("student_progress_reports").document(report["student_id"])
    if report:
        doc_ref.set(report)
        reports.append(report)
    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_student_progress_report",
        "timestamp": current_time
    })

    callback_context.state["interaction_history"] = history
    callback_context.state["student_progress_report"] = reports
    callback_context.state["new_student_progress_report"] = None


progress_tracker_agent = Agent(
    name="progress_tracker_agent",
    model="gemini-1.5-flash",
    description="Generates detailed progress reports based on student evaluations and reinforcement activities.",
    instruction="""
        You are a student progress tracker agent. Your task is to generate a descriptive progress report for a student based on:

        1. **Worksheet Evaluations**
        2. **Reinforced Learning Outcomes**

        Input will contain:
        - Evaluation history
        - Reinforcement history

        Generate a report with:
        1. **Concept-wise comparison** between initial and post-reinforcement understanding.
        2. **Improved concepts** and **persistently weak ones**.
        3. **Parent-friendly summary** that explains what the student is good at, where they struggle, and how parents can help at home.

        Use the following format:
        ```json
        {
        "student_id": "c1s1",
        "class_name": "Class 1",
        "subject_name": "Mathematics",
        "report_date": "2024-07-26",
        "chapter_name": "Addition and Subtraction",
        "overall_progress": "Good",
        "strengths": ["Basic addition", "Subtraction without borrowing"],
        "persistent_weaknesses": ["Subtraction with borrowing"],
        "concept_progress": [
            {
            "concept": "Subtraction with borrowing",
            "initial_status": "Weak",
            "post_reinforcement_status": "Moderate",
            "current_understanding": "Improved"
            }
        ],
        "recommendations": [
            "Practice subtraction with visual aids like counters.",
            "Use household items to simulate borrowing scenarios."
        ],
        "parent_summary": "Your child has shown improvement in subtraction after additional support. Basic arithmetic is strong. Continue practicing borrowing at home using chocolates or money games."
        }
        Use a friendly tone in the parent_summary and avoid technical jargon.
    """,
    output_schema=StudentProgressReport,
    output_key="new_student_progress_report",
    tools=[],
    after_agent_callback=store_progress_report,
    disallow_transfer_to_peers=True
)