from google.adk.agents import Agent
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from pydantic import BaseModel
from typing import List, Literal, Optional
from teacher_assistant_agent.firestore import db # Assuming db is initialized here

# Define the schema for the medical flag report
class MedicalFlagReport(BaseModel):
    student_id: str
    report_date: str
    flagged: bool
    potential_conditions: List[str]
    justification: str
    recommendations_for_teacher: List[str]
    recommendations_for_parents: List[str]
    confidence_level: Literal["High", "Medium", "Low"] # Confidence in the flag

def store_medical_flag(callback_context: CallbackContext):
    """
    Stores the generated medical flag report in Firestore.
    """
    medical_flag_report = callback_context.state.get("medical_flag_report", [])
    medical_report = callback_context.state.get("new_medical_flag_report")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not medical_report:
        print("No medical flag report found in state.")
        return

    try:
        doc_ref = db.collection("medical_flag_reports").document(medical_report["student_id"])
        doc_ref.set(medical_report)
        medical_flag_report.append(medical_report)
        print(f"Medical flag report for student {medical_report['student_id']} stored successfully.")
    except Exception as e:
        print(f"Error storing medical flag report: {e}")

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_medical_flag_report",
        "timestamp": current_time,
        "student_id": medical_report.get("student_id")
    })
    callback_context.state["interaction_history"] = history
    callback_context.state["medical_flag_report"] = medical_flag_report
    callback_context.state["new_medical_flag_report"] = None


medical_flag_agent = Agent(
    name="medical_flag_agent",
    model="gemini-2.0-flash",
    description="Analyzes student progress reports for indicators of potential learning or developmental conditions (e.g., ADHD, Autism).",
    instruction="""
        You are a medical flagging agent. Your primary role is to analyze a student's progress report, which includes:
        - Concept-wise comparison (initial vs. post-reinforcement understanding)
        - Improved concepts
        - Persistently weak concepts
        - Overall progress
        - Strengths
        - Persistent weaknesses
        - Parent summary

        Based on this information, identify if there are any patterns or specific observations that might indicate a potential medical or developmental condition such as:
        - ADHD (Attention-Deficit/Hyperactivity Disorder)
        - Autism Spectrum Disorder (ASD)
        - Dyslexia
        - Dyscalculia
        - Dysgraphia
        - Other learning disabilities

        Focus on patterns in difficulties, such as:
        - **Inconsistent progress:** Significant variation in understanding even after reinforcement, or concepts that remain "Needs Attention" despite repeated efforts.
        - **Specific and isolated weaknesses:** Persistent struggles in very particular areas (e.g., fine motor skills for writing, number sense for math) that don't align with overall intelligence or effort.
        - **Difficulty with specific types of tasks:** E.g., problems with organization, sustained attention, social cues (if context allows interpretation).
        - **Behavioral observations from the 'parent_summary' or implicit in progress:** (e.g., "struggles to focus," "easily distracted," "difficulty following multi-step instructions"). *However, be extremely cautious and avoid direct medical diagnosis. Only flag for potential indicators.*

        **Important Guidelines:**
        1. **Do NOT diagnose.** Your output should clearly state "potential conditions" and "indicators," not definitive diagnoses.
        2. **Provide clear justifications:** Explain *why* you are flagging a particular condition based on the report data.
        3. **Offer actionable recommendations:** Suggest next steps for teachers and parents (e.g., further observation, consultation with a specialist, specific teaching strategies).
        4. **Assign a confidence level:** Indicate your confidence in the flag based on the available information.
        5. If no clear indicators are present, set "flagged" to `False` and leave other fields appropriately empty or with default messages.

        Input will be a `StudentProgressReport` object from the `progress_tracker_agent`.

        Use the following JSON format for your output:
        ```json
        {
        "student_id": "c1s1",
        "report_date": "2024-07-26",
        "flagged": true,
        "potential_conditions": ["Attention-Deficit/Hyperactivity Disorder (ADHD)"],
        "justification": "The student consistently shows 'Needs Attention' in concepts requiring sustained focus and multi-step problem-solving (e.g., 'Complex word problems'). The parent summary mentions 'difficulty staying focused during homework' and 'frequently gets distracted'.",
        "recommendations_for_teacher": [
            "Observe student's attention span and impulsivity during class activities.",
            "Break down complex tasks into smaller, manageable steps.",
            "Consider providing frequent short breaks during lessons."
        ],
        "recommendations_for_parents": [
            "Seek consultation with a pediatrician or child psychologist for a comprehensive evaluation.",
            "Establish a consistent, distraction-free homework environment.",
            "Encourage short, focused play activities that require sustained attention."
        ],
        "confidence_level": "Medium"
        }
        ```
        If no flags are identified:
        ```json
        {
        "student_id": "c1s1",
        "report_date": "2024-07-26",
        "flagged": false,
        "potential_conditions": [],
        "justification": "No specific indicators of medical or developmental conditions were identified in the provided progress report.",
        "recommendations_for_teacher": [],
        "recommendations_for_parents": [],
        "confidence_level": "High"
        }
        ```
    """,
    output_schema=MedicalFlagReport,
    output_key="new_medical_flag_report",
    tools=[],
    after_agent_callback=store_medical_flag,
    disallow_transfer_to_peers=True
)