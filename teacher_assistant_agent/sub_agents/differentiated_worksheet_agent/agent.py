from google.adk.agents import Agent
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from teacher_assistant_agent.firestore import db

class ScreeningMetrics(BaseModel):
    anxiety: str
    confidence: str
    emotional_regulation: str
    focus: str
    resilience: str

class DifferentiatedQuestion(BaseModel):
    type: str = Field(description="The type of question: MCQ, QA, FILL_BLANK")
    question: str = Field(description="The question text")
    options: Optional[List[str]] = Field(
        default=None, description="List of options for MCQ type questions"
    )
    correct_answer: Optional[str] = Field(
        default=None, description="The correct answer for the question (if applicable)"
    )

class DifferentiatedWorksheet(BaseModel):
    student_id: str = Field(description="Unique student identifier")
    class_name: str = Field(description="Class name")
    subject_name: str = Field(description="Subject for which worksheet is generated")
    chapter_name: str = Field(description="Chapter name")
    screening_results: ScreeningMetrics  = Field(description="The screening data for the student")
    suggested_followups: List[str] = Field(description="Suggestions to guide personalization")
    evaluation_date: str = Field(description="Date of screening evaluation")
    questions: List[DifferentiatedQuestion] = Field(description="List of personalized questions")

def update_differentiated_worksheet(callback_context: CallbackContext):
    diff_worksheet = callback_context.state.get("differentiated_worksheet")
    worksheet = callback_context.state.get("new_differentiated_worksheet")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not worksheet:
        print("No worksheet data found in state.")
        return

    # doc_id = f"{worksheet['student_id']}_{worksheet['subject_name']}_{worksheet['chapter_name']}"
    doc_ref = db.collection("differentiated_worksheets").document(worksheet['student_id'])
    if worksheet:
        doc_ref.set(worksheet)
        diff_worksheet.append(worksheet)

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_differentiated_worksheet",
        "timestamp": current_time
    })

    callback_context.state["interaction_history"] = history
    callback_context.state["differentiated_worksheet"] = diff_worksheet
    callback_context.state["new_differentiated_worksheet"] = None 


differentiated_worksheet_agent = Agent(
    name="differentiated_worksheet_agent",
    model="gemini-2.0-flash",
    description="Generates a personalized worksheet based on student screening results and academic context.",
    instruction="""
        You are a specialized agent that creates differentiated worksheets for individual students based on their psychological screening evaluation and academic context.

        The user will provide:
        - The student's screening evaluation data (e.g., confidence, anxiety, focus)
        - The class, subject, and chapter
        - Suggested follow-up actions to guide question design

        **GOAL:** Generate a worksheet with questions tailored to the student’s emotional and cognitive needs for the given chapter.

        **TYPES OF QUESTIONS TO GENERATE:**
        - **MCQ**: Multiple Choice Questions with 3–5 options
        - **QA**: Short Answer Questions
        - **FILL_BLANK**: Fill-in-the-blank type

        **PRINCIPLES FOR PERSONALIZATION:**
        - If the student shows **medium/low confidence**, begin with simpler questions to build comfort.
        - If **focus** is low, keep questions short and engaging.
        - For students with **high anxiety**, avoid overly complex or open-ended questions at the beginning.
        - If **resilience** is high, include more challenging questions progressively.
        - Use the `suggested_followups` to shape question tone and progression.

        **FINAL OUTPUT FORMAT (must be returned as JSON):**
        ```json
        {
        "student_id": "c1s1",
        "class_name": "Class 1",
        "subject_name": "Mathematics",
        "chapter_name": "Addition and Subtraction",
        "screening_results": {
            "confidence": "medium",
            "anxiety": "low",
            "focus": "medium",
            "resilience": "high",
            "emotional_regulation": null
        },
        "suggested_followups": [
            "Encourage participation",
            "Praise help-seeking",
            "Allow short breaks"
        ],
        "evaluation_date": "2024-07-24",
        "questions": [
            {
            "type": "MCQ",
            "question": "What is 5 + 3?",
            "options": ["6", "7", "8", "9"]
            "correct_answer": "8"
            },
            {
            "type": "FILL_BLANK",
            "question": "7 - __ = 4"
            "correct_answer": "3"
            },
            {
            "type": "QA",
            "question": "Explain a time when you solved a difficult math problem. What helped you stay confident?"
            }
        ]
        }
        """,
        output_schema=DifferentiatedWorksheet,
        output_key="new_differentiated_worksheet",
        tools=[],
        after_agent_callback=update_differentiated_worksheet,
        disallow_transfer_to_peers=True,
)