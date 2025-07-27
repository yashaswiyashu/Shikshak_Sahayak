from google.adk.agents import Agent
from teacher_assistant_agent.firestore import db
from datetime import datetime
from google.adk.agents.callback_context import CallbackContext
from pydantic import BaseModel, Field
from typing import List

class Topic(BaseModel):
    title: str = Field(description="Topic title")
    time_minutes: int = Field(description="Time in minutes")
    activity: str = Field(description="Activity name")

class DailyPlan(BaseModel):
    day: int = Field(description="Day number")
    title: str = Field(description="Title for the day")
    topics: List[Topic] = Field(description="List of topics covered")
    time_allocated_minutes: int = Field(description="Time allocated for the day in minutes")

class LessonPlan(BaseModel):
    teacher: str = Field(description="Name of the teacher")
    class_name: str = Field(description="Class name or grade")
    subject_name: str = Field(description="Subject name")
    chapter_name: str = Field(description="Chapter name")
    time_per_day_minutes: int = Field(description="Total time allocated per day in minutes")
    number_of_days: int = Field(description="Number of days for the chapter")
    short_description: str = Field(description="Short description of the chapter")
    learning_objective: str = Field(description="Learning objective of the chapter")
    daily_plan: List[DailyPlan] = Field(description="List of daily lesson plans")


def update_lesson_plan(callback_context: CallbackContext):
    lesson_plans = callback_context.state.get("lesson_plans", [])
    lesson_plan_data = callback_context.state.get("new_lesson_plan")
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # print(f"DEBUG - Storing lesson plan at {current_time}")
    print(f"DEBUG - Lesson plan data: {lesson_plan_data}")
    
    if not lesson_plan_data:
        print("No lesson plan data found in state.")
        return

    print(f"Saving lesson plan: {lesson_plan_data}")

    # doc_id = f"{lesson_plan_data['teacher']}_{lesson_plan_data['class_name']}_{lesson_plan_data['subject_name']}_{lesson_plan_data['chapter_name']}"
    doc_ref = db.collection("lesson_plans").document(lesson_plan_data['chapter_name'])
    if lesson_plan_data:
        doc_ref.set(lesson_plan_data)
        lesson_plans.append(lesson_plan_data)

    callback_context.state["lesson_plans"] = lesson_plans
    callback_context.state["new_psych_profile"] = None 

    history = callback_context.state.get("interaction_history", [])
    history.append({
        "action": "store_lesson_plan",
        "timestamp": current_time
    })
    callback_context.state["interaction_history"] = history
    callback_context.state["new_lesson_plan"] = None


lesson_planner_agent = Agent(
    name="lesson_planner_agent",
    model="gemini-2.0-flash",
    description="Generates a structured daily lesson plan based on chapter, subject, and total time per day.",
    instruction="""
        You are a lesson planning assistant that helps teachers create structured lesson plans for a given subject, class, and chapter.

        **MANDATORY USER INPUT:**
        - Total time available per day (in minutes). If the user has not provided this, ask explicitly: "How many minutes per day do you want to allocate for this lesson plan?"

        **STRUCTURE RULES:**
        1. Plan must include number of days (`number_of_days`) needed to complete the chapter.
        2. Each day will have:
            - A `title` for the day.
            - Topics list with:
                - Title
                - Estimated time in minutes
                - Activity type
        3. Distribute the `time_per_day_minutes` across `topics` within each day.
        4. The total time in `topics` should equal `time_allocated_minutes` for that day.

        **FINAL OUTPUT FORMAT (MUST be returned as JSON):**
        {
            "teacher": "Teacher Name",
            "class_name": "Class",
            "subject_name": "Subject",
            "chapter_name": "Chapter",
            "time_per_day_minutes": 45,
            "number_of_days": 5,
            "short_description": "Short summary of the chapter",
            "learning_objective": "What students will learn",
            "daily_plan": [
                {
                    "day": 1,
                    "title": "Introduction to the Chapter",
                    "topics": [
                        {
                            "title": "Introduction and key terms",
                            "time_minutes": 20,
                            "activity": "Discussion"
                        },
                        {
                            "title": "Overview of chapter themes",
                            "time_minutes": 25,
                            "activity": "Interactive lecture"
                        }
                    ],
                    "time_allocated_minutes": 45
                }
            ]
        }
    """,
    output_schema=LessonPlan,
    output_key="new_lesson_plan",
    tools=[],
    after_agent_callback=update_lesson_plan,
    disallow_transfer_to_peers=True,
)
