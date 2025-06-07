'''
module: app.py

This module defines a Gradio-based web interface for PCB defect detection, repair plan
generation, and interactive Q&A. It wires up the PCBDetector and RepairPlanGenerator
classes to UI components and state management.
'''
import gradio as gr
import os
from detect import PCBDetector
from plan import RepairPlanGenerator
import tempfile
import shutil

# Initialize PCB detection and planning components
detector = PCBDetector(model_path="models/best.onnx")
planner = RepairPlanGenerator(
    local_model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    use_openai=False
)

# Global state for last detection results and image paths
last_raw_detections = []
last_input_image_path = ""
last_annotated_image_path = ""

def run_detection(image):
    """
    Handle image upload, perform defect detection, and return an annotated image
    along with a textual summary of detected defects.

    :param image: PIL Image uploaded by the user via Gradio.
    :return: Tuple of (annotated_image, summary_text).
    """
    global last_raw_detections, last_input_image_path, last_annotated_image_path
    if image is None:
        return None, "Please upload an image."

    # Save the original and annotated images to temporary files for later use
    input_img_path = os.path.join(tempfile.gettempdir(), "input.png")
    annotated_img_path = os.path.join(tempfile.gettempdir(), "annotated.png")
    image.save(input_img_path)

    annotated_img, detections, summary = detector.detect_defects(image)
    annotated_img.save(annotated_img_path)

    # Update global state
    last_input_image_path = input_img_path
    last_annotated_image_path = annotated_img_path
    last_raw_detections = detections

    return annotated_img, summary


def toggle_openai(use_openai):
    """
    Switch the repair planner between local Llama model and OpenAI API modes.

    :param use_openai: Boolean flag from the Gradio checkbox.
    :return: Status message indicating the active model.
    """
    planner.use_openai = use_openai
    return "Using OpenAI API" if use_openai else "Using local Mistral-7B model"


def set_openai_key(api_key):
    """
    Validate and set the OpenAI API key for online inference.

    :param api_key: String provided by the user.
    :return: Success or error message.
    """
    if api_key:
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            client.models.list()  # test API connectivity
            os.environ["OPENAI_API_KEY"] = api_key
            planner.openai = client
            planner.use_openai = True
            return "✅ OpenAI API key set successfully. Using OpenAI API."
        except Exception as e:
            return f"❌ Invalid API key or connection error: {e}"
    return "❌ No API key provided."


def run_plan(defect_summary):
    """
    Generate a repair plan based on the detected defects summary.

    :param defect_summary: Text summary of defects from run_detection.
    :return: Tuple containing the plan Markdown, raw plan text, chat reset,
             the original summary, and token usage statistics.
    """
    if not last_raw_detections:
        return (
            "❌ No defects detected. Please run detection first.",
            "", [], defect_summary, ""
        )
    plan_text, token_summary = planner.generate_plan(
        defect_summary,
        raw_detections=last_raw_detections
    )
    return plan_text, plan_text, [], defect_summary, token_summary


def answer_question(user_message, history, plan_text, defects_summary):
    """
    Handle user queries in the chatbot tab, providing contextual answers based on
    the generated repair plan and optional images.

    :param user_message: The question entered by the user.
    :param history: Current chat history as a list of [user, assistant] pairs.
    :param plan_text: Text of the repair plan previously generated.
    :param defects_summary: Original defects summary text.
    :return: Updated history and cleared input box.
    """
    if history is None:
        history = []
    if not plan_text.strip():
        return history + [[user_message, "No repair plan available. Please generate one."]], ""

    image_paths = (
        [last_input_image_path, last_annotated_image_path]
        if planner.use_openai else None
    )
    answer = planner.answer_question(
        user_message,
        plan_text,
        chat_history=history,
        defects_info=defects_summary,
        image_paths=image_paths
    )
    return history + [[user_message, answer]], ""


def clear_chat():
    """
    Clear the chatbot conversation history.

    :return: Tuple of (empty history list, empty input string).
    """
    return [], ""

# Build and launch the Gradio application
demo = gr.Blocks(theme=gr.themes.Soft())
with demo:
    gr.Markdown("## PCB Defect Detection and Repair Planner")
    plan_state = gr.State("")
    defects_state = gr.State("")

    with gr.Tabs():
        with gr.Tab("Detection & Plan"):
            with gr.Row():
                openai_checkbox = gr.Checkbox(label="Use OpenAI API", value=False)
                openai_key_input = gr.Textbox(label="Enter OpenAI API Key", type="password")
                confirm_key_btn = gr.Button("Confirm API Key")
                model_status = gr.Textbox(
                    label="Model Status",
                    value="Using local Mistral-7B model",
                    interactive=False
                )

            with gr.Row():
                with gr.Column(scale=1):
                    image_input = gr.Image(label="PCB Image", type="pil")
                    detect_button = gr.Button("Detect Defects")
                    plan_button = gr.Button("Generate Repair Plan")
                with gr.Column(scale=1):
                    image_output = gr.Image(label="Image with Detected Defects", interactive=False)
                    defects_text = gr.Textbox(label="Defects Summary", interactive=False)
                    plan_output = gr.Markdown("**Repair Plan:** _not generated yet._")

            token_usage_display = gr.Textbox(label="OpenAI Token Usage", interactive=False)

        with gr.Tab("Chatbot Q&A"):
            chatbot = gr.Chatbot(label="Repair Plan Assistant")
            with gr.Row():
                user_input = gr.Textbox(label="Your Question", lines=2)
                send_button = gr.Button("Send")
                clear_button = gr.Button("Clear Chat")

    # Wire up event handlers
    openai_checkbox.change(
        fn=toggle_openai, inputs=openai_checkbox, outputs=model_status
    )
    confirm_key_btn.click(
        fn=set_openai_key, inputs=openai_key_input, outputs=model_status
    )
    detect_button.click(
        fn=run_detection, inputs=image_input, outputs=[image_output, defects_text]
    )
    plan_button.click(
        fn=run_plan,
        inputs=defects_text,
        outputs=[plan_output, plan_state, chatbot, defects_state, token_usage_display]
    )
    send_button.click(
        fn=answer_question,
        inputs=[user_input, chatbot, plan_state, defects_state],
        outputs=[chatbot, user_input]
    )
    user_input.submit(
        fn=answer_question,
        inputs=[user_input, chatbot, plan_state, defects_state],
        outputs=[chatbot, user_input]
    )
    clear_button.click(
        fn=clear_chat,
        outputs=[chatbot, user_input]
    )

    demo.launch()
