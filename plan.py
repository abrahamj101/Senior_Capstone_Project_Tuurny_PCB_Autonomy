'''
module: plan.py

This module defines RepairPlanGenerator, which creates detailed repair plans for
PCB defects using either a local Llama model or the OpenAI API, and logs token usage.
'''
import os
import re
from typing import List, Tuple, Optional
from datetime import datetime
import csv
import base64


class RepairPlanGenerator:
    """
    Generates step-by-step repair plans for PCB defects, leveraging a local
    Llama model or the OpenAI API.
    """
    def __init__(
        self,
        local_model_path: str = "models/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        use_openai: Optional[bool] = None,
        openai_model: str = "gpt-4o"
    ):
        """
        Initialize the RepairPlanGenerator.

        :param local_model_path: Path to local GGUF model file.
        :param use_openai: If None, auto-detect via OPENAI_API_KEY env var.
        :param openai_model: Model name to use when OpenAI API is enabled.
        """
        if use_openai is None:
            use_openai = bool(os.getenv("OPENAI_API_KEY"))
        self.use_openai = use_openai
        self.openai_model = openai_model
        self.local_model_path = local_model_path
        self.llm = None
        self.last_defects = None

        if self.use_openai:
            import openai
            from openai import OpenAI
            self.openai = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        else:
            from llama_cpp import Llama
            self.Llama = Llama

    def _load_local_model(self) -> None:
        """
        Load the local Llama model if not already loaded into memory.
        """
        if self.llm is None:
            self.llm = self.Llama(
                model_path=self.local_model_path,
                n_ctx=4096,
                n_threads=8
            )

    def _format_steps(self, text: str) -> str:
        """
        Insert newlines before each 'Step N:' in the generated text for readability.

        :param text: Raw plan text from LLM.
        :return: Formatted text with line breaks.
        """
        return re.sub(r'(?<!\n)(Step \d+:)', r'\n\1', text)

    def _log_token_usage(
        self,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        source: str = "repair_plan"
    ) -> None:
        """
        Append a record of token usage to a CSV logfile.

        :param prompt_tokens: Tokens sent in the prompt.
        :param completion_tokens: Tokens returned by the LLM.
        :param total_tokens: Sum of prompt and completion tokens.
        :param source: Context identifier for the log entry.
        """
        log_path = "token_usage_log.csv"
        timestamp = datetime.now().isoformat()
        file_exists = os.path.isfile(log_path)
        with open(log_path, "a", newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "Timestamp",
                    "Prompt Tokens",
                    "Completion Tokens",
                    "Total Tokens",
                    "Source"
                ])
            writer.writerow([timestamp, prompt_tokens, completion_tokens, total_tokens, source])

    def generate_plan(
        self,
        defects_summary: str,
        raw_detections: Optional[List[dict]] = None
    ) -> Tuple[str, str]:
        """
        Generate a combined repair plan for all defect types or a single summary.

        :param defects_summary: Text summary of detected defects.
        :param raw_detections: List of dicts with individual detection details.
        :return: Markdown-formatted plan and token usage summary.
        """
        self.last_defects = defects_summary
        if raw_detections:
            defect_types = {}
            for det in raw_detections:
                defect_types.setdefault(det["class"], []).append(det)
            all_plans = []
            usage = ""
            for dtype, items in defect_types.items():
                count = len(items)
                summary = f"{count}x {dtype}"
                subplan, usage = self._generate_single_plan(summary)
                all_plans.append(f"## Repair Plan for {dtype} ({count} instances)\n\n{subplan}")
            return "# PCB Repair Plan\n\n" + "\n\n".join(all_plans), usage
        return self._generate_single_plan(defects_summary)

    def _generate_single_plan(
        self,
        defect_description: str
    ) -> Tuple[str, str]:
        """
        Create a step-by-step repair plan for a single defect description.

        :param defect_description: Brief description (e.g., '3x open circuit').
        :return: Tuple of (formatted plan text, token usage summary).
        """
        prompt = (
            f"You are a PCB repair expert. A PCB has the following issues:\n"
            f"{defect_description}\n"
            "Provide a detailed step-by-step plan to repair these defects. "
            "Include specific tools, techniques, and safety precautions. "
            "Format your response as a numbered list of steps."
        )
        if self.use_openai:
            messages = [
                {"role": "system", "content": "You are a helpful assistant specialized in PCB repair."},
                {"role": "user", "content": prompt}
            ]
            try:
                resp = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=600
                )
                plan_text = resp.choices[0].message.content.strip()
                usage = resp.usage.model_dump() if hasattr(resp, "usage") else {}
                self._log_token_usage(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0)
                )
                return self._format_steps(plan_text), (
                    f"Tokens — Prompt: {usage.get('prompt_tokens')}, "
                    f"Completion: {usage.get('completion_tokens')}, "
                    f"Total: {usage.get('total_tokens')}"
                )
            except Exception as e:
                return f"Error generating plan: {e}", ""
        # Local model path
        self._load_local_model()
        try:
            result = self.llm.create_completion(
                prompt=prompt,
                max_tokens=600,
                temperature=0.5,
                stop=["\nUser", "\nAssistant"]
            )
            plan_text = result.get("choices", [{"text": str(result)}])[0]["text"].strip()
            return self._format_steps(plan_text), ""
        except Exception as e:
            return f"Error generating plan: {e}", ""

    def _encode_image_to_base64(self, path: str) -> str:
        """
        Encode an image file as a base64 string for embedding in API requests.

        :param path: Path to image file.
        :return: Base64-encoded string.
        """
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def answer_question(
        self,
        question: str,
        plan_text: str,
        chat_history: Optional[List[Tuple[str, str]]] = None,
        defects_info: Optional[str] = None,
        image_paths: Optional[List[str]] = None
    ) -> str:
        """
        Answer a user question based on the existing repair plan context.

        :param question: User question to answer.
        :param plan_text: The generated repair plan text.
        :param chat_history: Previous conversation history.
        :param defects_info: Original defects summary.
        :param image_paths: List of image file paths for context.
        :return: The assistant's response string.
        """
        chat_history = chat_history or []
        defects_info = defects_info or self.last_defects
        context = f"Repair Plan:\n{plan_text}\n\n"
        if defects_info:
            context += f"Detected Defects:\n{defects_info}\n\n"
        if self.use_openai:
            messages = [{"role": "system", "content": f"You are a PCB repair assistant.\n{context}"}]
            if image_paths:
                for img in image_paths:
                    b64 = self._encode_image_to_base64(img)
                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "text", "text": f"This image is relevant to the question:\n{question}"},
                            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
                        ]
                    })
            for usr, ans in chat_history:
                messages.append({"role": "user", "content": usr})
                messages.append({"role": "assistant", "content": ans})
            messages.append({"role": "user", "content": question})
            try:
                resp = self.openai.chat.completions.create(
                    model=self.openai_model,
                    messages=messages,
                    temperature=0.5,
                    max_tokens=600
                )
                answer = resp.choices[0].message.content.strip()
                usage = resp.usage.model_dump() if hasattr(resp, "usage") else {}
                self._log_token_usage(
                    usage.get("prompt_tokens", 0),
                    usage.get("completion_tokens", 0),
                    usage.get("total_tokens", 0),
                    source="chat"
                )
                return answer
            except Exception as e:
                return f"Error: {e}"
        return "Error: OpenAI integration disabled."