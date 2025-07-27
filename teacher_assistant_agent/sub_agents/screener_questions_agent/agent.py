from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from typing import List, Optional
from pydantic import BaseModel, Field
from google.adk.tools.tool_context import ToolContext
from google.adk.agents.callback_context import CallbackContext
from datetime import datetime
from teacher_assistant_agent.firestore import db


class Question(BaseModel):
    """Represents a single psychological screening question."""
    type: str = Field(description="The type of question (MCQ, QA, FILL_BLANK).")
    question: str = Field(description="The text of the question.")
    options: Optional[List[str]] = Field(
        default=None, description="List of options for MCQ type questions."
    )

class QuestionSet(BaseModel):
    """Represents a set of psychological screening questions."""
    question_set_title: str = Field(description="The title of the question set.")
    questions: List[Question] = Field(description="A list of questions in the set.")

def update_questions_set(callback_context: CallbackContext): # Removed the return type hint as it's not expected to return a structured dict
    """
    Update the existing questions set with new questions.
    
    Args:
        callback_context (CallbackContext): The context object containing agent state.
    """

    # doc_ref = db.collection("users").document("alovelace")
    # doc_ref.set({"first": "Ada", "last": "Lovelace", "born": 1815})
    # print(f"{callback_context.state} Calling update_questions_set")
    questions_set = callback_context.state.get("questions_set", []) # Initialize as empty list if not present
    new_questions_set_data = callback_context.state.get("new_questions_set")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"Current questions set: {questions_set}")
    print(f"New questions to add: {new_questions_set_data}")

    doc_ref = db.collection("questions_set").document(f"{new_questions_set_data["question_set_title"]}")
    if new_questions_set_data:
        # Ensure new_questions_set_data is a QuestionSet object or convert it
        # Since output_key="new_questions_set" stores the Pydantic object, we can append directly
        doc_ref.set(new_questions_set_data)
        questions_set.append(new_questions_set_data)
    
    callback_context.state["questions_set"] = questions_set
    callback_context.state["new_questions_set"] = None  # Clear after updating
    print(f"Updated questions set: {questions_set}")
    
    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_question_set",
        "timestamp": current_time
    })
    callback_context.state["interaction_history"] = history
    # The after_agent_callback should not return values that the framework
    # tries to validate as events or agent output. Its purpose is to
    # perform side effects, like updating the state.
    # Therefore, remove the return statement.
    # return {
    #     "status": "success",
    #     "message": f"Question set stored successfully.",
    #     "timestamp": current_time
    # }

screener_questions_agent = Agent(
    name="screener_questions_agent",
    model="gemini-2.0-flash",
    description="An AI agent that generates psychological screening questions for students based on class and age.",
    instruction="""You are a specialized agent that generates psychological screening questions for students based on their class and age
    
        You will help user in generating psychological screening questions for students in a specific class and age group.
        
        Use `research_agent` tool to find relevant information about psychological screening questions.

        **GUIDELINES FOR GENERATING QUESTIONS:**
        1. Generate questions based on the class level provided (e.g., Grade 6, Grade 9)..
        2. Mix the following question formats:
            - **MCQ** (Multiple Choice Questions): Provide a list of options.
            - **QA** (Short Descriptive Answers): Open-ended questions.
            - **FILL_BLANK** (Fill-in-the-blank): A sentence with a blank.
        3. Focus on general mental well-being, social-emotional learning, and common developmental aspects, NOT subject-specific content.
        4. Ensure questions are insightful, supportive, and non-judgmental.

         IMPORTANT: Always return your response as a JSON object with the following structure:
        {
            "question_set_title": "Descriptive title for the question set",
            "questions": [
                {
                "type": "Question format (MCQ/QA/FILL_BLANK)",
                "question": "The question text",
                "options": ["Only for MCQ", "List of choices"] 
                }
            ]
        }
        **After generating the question set, acknowledge that the questions have been generated and inform the user they have been saved.
    """,
    output_schema=QuestionSet,
    output_key="new_questions_set",
    tools=[],
    after_agent_callback=update_questions_set,
    disallow_transfer_to_peers=True,
)