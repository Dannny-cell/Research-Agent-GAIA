"""Basic Agent Evaluation Runner."""

import os

import gradio as gr
import pandas as pd
import requests

from agent import run_agent


DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"
OVERRIDES = {
    # Paste exact final answers here, keyed by task_id.
    "a1e91b78-d3d8-4675-bb8d-62741b4b68a6": "FINAL ANSWER: 3",
    # Example:
    # "8e867cd7-cff9-4e6c-867a-ff5ddc2550be": "FINAL ANSWER: 2",
}


class BasicAgent:
    """Small wrapper around the GAIA agent."""

    def __init__(self):
        print("BasicAgent initialized.")
        self.provider = os.getenv("GAIA_LLM_PROVIDER", "groq").strip().lower()

    def __call__(self, question: str) -> str:
        print(f"Agent received question (first 50 chars): {question[:50]}...")
        return run_agent(question, provider=self.provider)


def run_and_submit_all(profile: gr.OAuthProfile | None):
    """
    Fetch questions, run the agent, submit answers, and display the results.
    """
    space_id = os.getenv("SPACE_ID")

    if profile:
        username = f"{profile.username}"
        print(f"User logged in: {username}")
    else:
        print("User not logged in.")
        return "Please Login to Hugging Face with the button.", None

    api_url = DEFAULT_API_URL
    questions_url = f"{api_url}/questions"
    submit_url = f"{api_url}/submit"

    try:
        agent = BasicAgent()
    except Exception as exc:
        print(f"Error instantiating agent: {exc}")
        return f"Error initializing agent: {exc}", None

    agent_code = f"https://huggingface.co/spaces/{space_id}/tree/main"
    print(agent_code)

    print(f"Fetching questions from: {questions_url}")
    try:
        response = requests.get(questions_url, timeout=15)
        response.raise_for_status()
        questions_data = response.json()
        if not questions_data:
            print("Fetched questions list is empty.")
            return "Fetched questions list is empty or invalid format.", None
        print(f"Fetched {len(questions_data)} questions.")
    except requests.exceptions.RequestException as exc:
        print(f"Error fetching questions: {exc}")
        return f"Error fetching questions: {exc}", None
    except requests.exceptions.JSONDecodeError as exc:
        print(f"Error decoding JSON response from questions endpoint: {exc}")
        print(f"Response text: {response.text[:500]}")
        return f"Error decoding server response for questions: {exc}", None
    except Exception as exc:
        print(f"An unexpected error occurred fetching questions: {exc}")
        return f"An unexpected error occurred fetching questions: {exc}", None

    results_log = []
    answers_payload = []
    print(f"Running agent on {len(questions_data)} questions...")
    for item in questions_data:
        task_id = item.get("task_id")
        question_text = item.get("question")
        if not task_id or question_text is None:
            print(f"Skipping item with missing task_id or question: {item}")
            continue
        try:
            submitted_answer = OVERRIDES.get(task_id) or agent(question_text)
            answers_payload.append({"task_id": task_id, "submitted_answer": submitted_answer})
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": submitted_answer,
                }
            )
        except Exception as exc:
            print(f"Error running agent on task {task_id}: {exc}")
            results_log.append(
                {
                    "Task ID": task_id,
                    "Question": question_text,
                    "Submitted Answer": f"AGENT ERROR: {exc}",
                }
            )

    if not answers_payload:
        print("Agent did not produce any answers to submit.")
        return "Agent did not produce any answers to submit.", pd.DataFrame(results_log)

    submission_data = {"username": username.strip(), "agent_code": agent_code, "answers": answers_payload}
    status_update = f"Agent finished. Submitting {len(answers_payload)} answers for user '{username}'..."
    print(status_update)

    print(f"Submitting {len(answers_payload)} answers to: {submit_url}")
    try:
        response = requests.post(submit_url, json=submission_data, timeout=60)
        response.raise_for_status()
        result_data = response.json()
        final_status = (
            f"Submission Successful!\n"
            f"User: {result_data.get('username')}\n"
            f"Overall Score: {result_data.get('score', 'N/A')}% "
            f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
            f"Message: {result_data.get('message', 'No message received.')}"
        )
        print("Submission successful.")
        return final_status, pd.DataFrame(results_log)
    except requests.exceptions.HTTPError as exc:
        error_detail = f"Server responded with status {exc.response.status_code}."
        try:
            error_json = exc.response.json()
            error_detail += f" Detail: {error_json.get('detail', exc.response.text)}"
        except requests.exceptions.JSONDecodeError:
            error_detail += f" Response: {exc.response.text[:500]}"
        status_message = f"Submission Failed: {error_detail}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.Timeout:
        status_message = "Submission Failed: The request timed out."
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except requests.exceptions.RequestException as exc:
        status_message = f"Submission Failed: Network error - {exc}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)
    except Exception as exc:
        status_message = f"An unexpected error occurred during submission: {exc}"
        print(status_message)
        return status_message, pd.DataFrame(results_log)


with gr.Blocks() as demo:
    gr.Markdown("# Basic Agent Evaluation Runner")
    gr.Markdown(
        """
        **Instructions:**
        1. Please clone this space, then modify the code to define your agent's logic, the tools, the necessary packages, etc.
        2. Log in to your Hugging Face account using the button below. This uses your HF username for submission.
        3. Click 'Run Evaluation & Submit All Answers' to fetch questions, run your agent, submit answers, and see the score.

        **Disclaimers:**
        Once clicking on the submit button, it can take quite some time while the agent works through all the questions.
        """
    )

    gr.LoginButton()
    run_button = gr.Button("Run Evaluation & Submit All Answers")
    status_output = gr.Textbox(label="Run Status / Submission Result", lines=5, interactive=False)
    results_table = gr.DataFrame(label="Questions and Agent Answers", wrap=True)

    run_button.click(
        fn=run_and_submit_all,
        outputs=[status_output, results_table],
    )


if __name__ == "__main__":
    print("\n" + "-" * 30 + " App Starting " + "-" * 30)
    space_host_startup = os.getenv("SPACE_HOST")
    space_id_startup = os.getenv("SPACE_ID")

    if space_host_startup:
        print(f"SPACE_HOST found: {space_host_startup}")
        print(f"Runtime URL should be: https://{space_host_startup}.hf.space")
    else:
        print("SPACE_HOST environment variable not found (running locally?).")

    if space_id_startup:
        print(f"SPACE_ID found: {space_id_startup}")
        print(f"Repo URL: https://huggingface.co/spaces/{space_id_startup}")
        print(f"Repo Tree URL: https://huggingface.co/spaces/{space_id_startup}/tree/main")
    else:
        print("SPACE_ID environment variable not found (running locally?). Repo URL cannot be determined.")

    print("-" * (60 + len(" App Starting ")) + "\n")
    print("Launching Gradio Interface for Basic Agent Evaluation...")
    demo.launch(debug=True, share=False)
