from google.adk.agents import Agent
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from teacher_assistant_agent.firestore import db

class ReinforcementQuestion(BaseModel):
    topic: str
    explanation: str
    analogy: Optional[str]
    question: str
    options: Optional[List[str]]
    correct_answer: str
    question_type: Literal["MCQ", "QA", "FILL_BLANK"]

class PersonalizedReinforcement(BaseModel):
    student_id: str
    subject_name: str
    chapter_name: str
    weak_areas: List[str]
    reinforcement_date: str
    reinforcement_questions: List[ReinforcementQuestion]

def store_reinforcement(callback_context: CallbackContext):
    personalized_reinforcement = callback_context.state.get("personalized_reinforcement")
    reinforcement = callback_context.state.get("new_personalized_reinforcement")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if not reinforcement:
        print("No reinforcement data found in state.")
        return

    doc_ref = db.collection("personalized_reinforcement").document(reinforcement["student_id"])
    if reinforcement:
      doc_ref.set(reinforcement)
      personalized_reinforcement.append(reinforcement)

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_personalized_reinforcement",
        "timestamp": current_time
    })

    callback_context.state["interaction_history"] = history
    callback_context.state["personalized_reinforcement"] = personalized_reinforcement
    callback_context.state["new_personalized_reinforcement"] = None


reinforcement_agent = Agent(
    name="reinforcement_agent",
    model="gemini-1.5-flash",
    description="Generates personalized and interactive re-learning sessions for students who need conceptual reinforcement based on their worksheet evaluations. It explains weak concepts using simple language and real-life analogies, then asks a follow-up question to check understanding. Helps identify and bridge learning gaps through engaging, tailored micro-lessons.",
    instruction="""
    You are a reinforcement learning agent helping students who are weak in specific topics.

    Input: A worksheet evaluation JSON in this format:
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
        },
        {
          "question": "What is 12 - 7?",
          "question_type": "QA",
          "is_correct": null,
          "feedback": "The explanation lacked clarity on subtraction with borrowing."
        }
      ]
    }

    Your tasks:

    For each weak area:

    Provide a simple explanation of the topic.

    Use a real-life analogy to make the concept relatable (like borrowing in money or chocolates).

    Ask one question (MCQ or fill-in-the-blank or QA) to test their understanding.

    Example json output:
    {
      "student_id": "c1s1",
      "subject_name": "Mathematics",
      "chapter_name": "Addition and Subtraction",
      "weak_areas": ["Subtraction with borrowing"],
      "reinforcement_date": "2024-07-26",
      "reinforcement_questions": [
        {
          "topic": "Subtraction with borrowing",
          "explanation": "Sometimes in subtraction, the top digit is smaller than the bottom digit, so we borrow from the next column.",
          "analogy": "Imagine you have 2 chocolates but need to give 5. You borrow 1 chocolate pack from your friend which has 10 pieces. Now you have 12 and can subtract.",
          "question": "What is 32 - 18?",
          "options": ["14", "12", "16", "10"],
          "correct_answer": "14",
          "question_type": "MCQ"
        }
      ]
    }
    Use simple language. Ensure that every explanation-question pair makes learning fun and easy.
  """,
  output_schema=PersonalizedReinforcement,
  output_key="new_personalized_reinforcement",
  tools=[],
  after_agent_callback=store_reinforcement,
  disallow_transfer_to_peers=True
)