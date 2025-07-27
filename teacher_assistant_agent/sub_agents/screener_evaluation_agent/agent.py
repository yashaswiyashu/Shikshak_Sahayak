from google.adk.agents import LlmAgent
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from datetime import datetime
from teacher_assistant_agent.firestore import db

class ScreeningResults(BaseModel):
    confidence: str = Field(description="Confidence level (e.g., 'low', 'medium', 'high').")
    anxiety: str = Field(description="Anxiety level (e.g., 'low', 'medium', 'high').")
    focus: str = Field(description="Focus level (e.g., 'low', 'medium', 'high').")
    resilience: str = Field(description="Resilience level (e.g., 'low', 'medium', 'high').")
    emotional_regulation: Optional[str] = Field(
        default=None, description="Emotional regulation level (optional)."
    )

class PsychProfileResult(BaseModel):
    student_id: str = Field(description="The unique identifier for the student.")
    class_name: str = Field(description="The class the student belongs to.")
    screening_results: ScreeningResults = Field(
        description="Detailed psychological screening results."
    )
    suggested_followups: List[str] = Field(
        description="List of suggested follow-up actions or recommendations."
    )
    evaluation_date: str = Field(description="Date of evaluation in YYYY-MM-DD format")

def store_psych_profile(callback_context: CallbackContext) -> dict:
    """
    Store the psych profile of a student in the teacher's shared state.
    Matches student by ID and updates their psych_profile field.
    """
    # Get the profile data from the agent's output
    profile_data = callback_context.state.get("new_psych_profile")
    student_id = profile_data.get("student_id")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"DEBUG - Storing profile for student {student_id} at {current_time}")
    print(f"DEBUG - Profile data: {profile_data}")

    # Get the current state
    psych_profile = callback_context.state.get("psych_profile", []) # Initialize as empty list if not present
    new_psyc_profile = callback_context.state.get("new_psych_profile")

    doc_ref = db.collection("screening_profile").document(f"{new_psyc_profile["student_id"]}")
    if new_psyc_profile:
        doc_ref.set(new_psyc_profile)
        psych_profile.append(new_psyc_profile)

    callback_context.state["psych_profile"] = psych_profile
    callback_context.state["new_psych_profile"] = None  # Clear after updating

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_profile_evaluation",
        "timestamp": current_time
    })
    callback_context.state["interaction_history"] = history


screener_evaluation_agent = LlmAgent(
    name="screener_evaluation_agent",
    model="gemini-2.0-flash", # You can keep flash here if it's just for text generation
    description="Evaluates student responses to psychological screenings and generates a psych profile.",
    instruction="""
        You are a profiling expert that:
            1. Analyzes student responses to psychological screenings
            2. Generates a psych profile with 5 key metrics
            3. Provides actionable follow-up recommendations
            4. Formats output for automatic state updates

        **TASK: EVALUATING STUDENT RESPONSES**

        Your task is to analyze student responses to psychological screening questions.
        Expect the input in the format:
        {
          "student_id": "S101",
          "class_name": "Class 6",
          "answers": [
            {"question": "How often do you feel nervous in a classroom?", "answer": "Sometimes"},
            {"question": "What do you do when you feel stressed before exams?", "answer": "I take deep breaths and listen to music."},
            {"question": "When I feel sad, I usually ________.", "answer": "talk to my parents."}
          ]
        }

        **GUIDELINES FOR EVALUATION:**
        1. Evaluate the answers for psychological markers such as:
            - **Confidence:** How self-assured the student appears.
            - **Anxiety:** Indicators of nervousness or worry.
            - **Focus:** Ability to concentrate and stay on task.
            - **Emotional Regulation:** How well they manage their feelings.
            - **Resilience:** Their ability to bounce back from difficulties.
        2. Assign a qualitative level (e.g., "low", "medium", "high") for each marker.
        3. Suggest actionable follow-up recommendations that are supportive and constructive.
        4. Your evaluation should be age-appropriate, insightful, and supportive — never judgmental.

        OUTPUT REQUIREMENTS:
            - Must be valid JSON matching PsychProfileResult schema
            - Include current date in evaluation_date
            - Follow-up suggestions should be specific and actionable
            - Never include explanatory text outside the JSON
                
        **IMPORTANT: Your final response MUST be a valid JSON object matching the `PsychProfileResult` schema.**
        DO NOT include any explanations or additional text outside the JSON response.
        You are not to call any tools directly. Your only job is to produce the JSON output.
        The parent agent (`teacher_assistant_agent`) will handle storing the output.
    """,
    output_schema=PsychProfileResult,
    output_key="new_psych_profile",
    tools=[], # FIX: This MUST be empty
    after_agent_callback=store_psych_profile,
    disallow_transfer_to_peers=True
)