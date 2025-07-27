from google.adk.agents import Agent
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from teacher_assistant_agent.firestore import db
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class AnswerFeedback(BaseModel):
    question: str
    question_type: str
    is_correct: Optional[bool]
    feedback: str

class EvaluationSummary(BaseModel):
    overall_understanding: Literal["Good", "Average", "Needs Improvement"]
    conceptual_strengths: List[str]
    conceptual_weaknesses: List[str]
    chapter_coverage: Literal["Fully covered", "Partially covered", "Poor"]
    suggested_retest_areas: List[str] | None


class WorksheetEvaluation(BaseModel):
    student_id: str
    class_name: str
    subject_name: str
    chapter_name: str
    evaluation_date: str
    summary: EvaluationSummary
    answer_feedback: List[AnswerFeedback]


def update_evaluation_result(callback_context: CallbackContext):
    worksheet_evaluation = callback_context.state.get("worksheet_evaluation", [])
    evaluation = callback_context.state.get("new_worksheet_evaluation")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not evaluation:
        print("No evaluation data found in state.")
        return

    # doc_id = f"{evaluation['student_id']}_{evaluation['subject_name']}_{evaluation['chapter_name']}_eval"
    doc_ref = db.collection("worksheet_evaluations").document(evaluation['student_id'])
    if evaluation:
        doc_ref.set(evaluation)
        worksheet_evaluation.append(evaluation)

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_worksheet_evaluation",
        "timestamp": current_time
    })

    callback_context.state["interaction_history"] = history
    callback_context.state["worksheet_evaluation"] = worksheet_evaluation
    callback_context.state["new_worksheet_evaluation"] = None

worksheet_evaluator_agent = Agent(
    name="worksheet_evaluator_agent",
    model="gemini-2.0-flash",
    description="Evalates student answers to a worksheet and provides analysis on understanding and concept mastery.",
    instruction="""
        You are an evaluator assistant that analyzes student-submitted worksheet answers.

        Given:
        - A list of questions
        - Student answers
        - Expected answers

        Your job:
        1. Compare student answer with the expected answer:
            Expect the input in the format:
            {
                "student_id": "c1s1",
                "class_name": "Class 1",
                "subject_name": "Mathematics",
                "chapter_name": "Addition and Subtraction",
                "evaluation_date": "2024-07-26",
                "answers": [
                    {
                    "question": "What is 5 + 3?",
                    "question_type": "MCQ",
                    "student_answer": "8",
                    "expected_answer": "8"
                    },
                    {
                    "question": "7 - __ = 4",
                    "question_type": "FILL_BLANK",
                    "student_answer": "3",
                    "expected_answer": "3"
                    },
                    {
                    "question": "Explain a time when you solved a difficult math problem. What helped you stay confident?",
                    "question_type": "QA",
                    "student_answer": "I asked my teacher and practiced more.",
                    "expected_answer": "Explanation showing help-seeking and persistence"
                    }
                ]
            }

            - For **MCQ** and **FILL_BLANK**, mark it correct or incorrect.
            - For **QA** (subjective), assess the **depth and relevance**. If the answer shows effort or aligns with expected reasoning, mark it with `is_correct: null` and give a thoughtful feedback.

        2. Provide detailed feedback for each question under `answer_feedback`.

        3. Create an overall evaluation `summary`:
            - `overall_understanding`: Good / Average / Needs Improvement
            - `conceptual_strengths`: List based on observed patterns
            - `conceptual_weaknesses`: Same as above
            - `chapter_coverage`: Fully covered / Partially covered / Poor
            - `suggested_retest_areas`: List concepts or topics where the student needs a retest (only if any conceptual weakness is detected)
        
        **Return JSON in the following format:**
        ```json
        {
        "student_id": "c1s1",
        "class_name": "Class 1",
        "subject_name": "Mathematics",
        "chapter_name": "Addition and Subtraction",
        "evaluation_date": "2024-07-26",
        "summary": {
            "overall_understanding": "Average",
            "conceptual_strengths": ["Basic addition"],
            "conceptual_weaknesses": ["Borrowing in subtraction"],
            "chapter_coverage": "Partially covered",
            "suggested_retest_areas": ["Subtraction with borrowing"]
        },
        "answer_feedback": [
            {
            "question": "What is 5 + 3?",
            "question_type": "MCQ",
            "is_correct": true,
            "feedback": "Correct answer."
            }
        ]
        }
    """,
    output_schema=WorksheetEvaluation,
    output_key="new_worksheet_evaluation",
    tools=[],
    after_agent_callback=update_evaluation_result,
    disallow_transfer_to_peers=True,
)
