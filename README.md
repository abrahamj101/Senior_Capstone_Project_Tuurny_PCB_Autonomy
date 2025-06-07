
# Full Project Documentation

## Project Overview

This project is a **PCB Defect Detection and Repair Planning** system.  
It uses **YOLOv8** for defect detection and a **Large Language Model (LLM)** (local or OpenAI GPT) to generate **step-by-step repair plans** based on detected defects.  
It also includes a **Gradio-based web app** for uploading images, detecting defects, generating repair plans, and chatting about repairs.

---

## Modules

# `plan.py — PCB Repair Plan Generator`

Handles the generation of **PCB defect repair plans** based on detected defects.  
Supports both:
- **Local LLM** (like Mistral-7B)
- **OpenAI GPT API** (if API key is provided).

Also answers user questions about the repair plan, optionally with images.

### Class: `RepairPlanGenerator`

**Constructor:**
```python
RepairPlanGenerator(local_model_path="models/mistral-7b-instruct-v0.1.Q4_K_M.gguf", use_openai=None, openai_model="gpt-4o")
```

**Key Methods:**
- `generate_plan(defects_summary, raw_detections=None)`: Create a repair plan based on defects detected.
- `answer_question(question, plan_text, chat_history=None, defects_info=None, image_paths=None)`: Answer follow-up questions about repair.
- Internal utilities for model loading, formatting steps, logging token usage, and image encoding.

---

# `detect.py — PCB Defect Detection`

Handles **detecting defects** on PCBs using a **YOLOv8 model**, and produces annotated results.

### Class: `PCBDetector`

**Constructor:**
```python
PCBDetector(model_path="models/best.onnx", device="cpu")
```

**Key Methods:**
- `load_model()`: Load YOLOv8 model.
- `detect_defects(image)`: Detect defects in a PIL image.
  - Outputs: Annotated image, list of detections, textual summary.

---

# `app.py — Gradio Web Application`

Defines a **Gradio app** that:
- Uploads PCB images
- Detects defects
- Generates repair plans
- Allows chatting about the repairs
- Supports toggling between local LLM and OpenAI

**Major Functions:**
- `run_detection(image)`: Runs defect detection.
- `toggle_openai(use_openai)`: Switch model backend.
- `set_openai_key(api_key)`: Set OpenAI API key.
- `run_plan(defect_summary)`: Generate repair plan.
- `answer_question(user_message, history, plan_text, defects_summary)`: Chat interaction.
- `clear_chat()`: Reset the chatbot.

---

## Gradio Interface Overview

**Tab 1:**  
- Upload PCB image
- Detect defects
- View annotated output
- View repair plan

**Tab 2:**  
- Chat with an assistant about repairs

---

# Project Structure

```
models/
    best.onnx          # YOLOv8 trained model
    mistral-7b-*.gguf  # Local LLM model
PCB_defect_detection/
    PCB_DATASET/
plan.py
detect.py
app.py
README.md
requirements.txt
```

---

# Setup

```bash
pip install -r requirements.txt
python app.py
```

---

# Summary Table

| Module    | Purpose                                  |
|-----------|------------------------------------------|
| plan.py   | Repair plan generation using LLM or GPT  |
| detect.py | PCB defect detection using YOLOv8        |
| app.py    | Gradio web app frontend                  |

---

# Quick Tips

- Models must exist under `models/` directory.
- To use OpenAI, provide your API key via the app UI.
- Trained models (`onnx`) should be exported after YOLOv8 training.
- Automatically logs token usage when using OpenAI.

---