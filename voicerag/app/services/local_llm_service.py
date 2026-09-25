import os
import sys
import re
from pathlib import Path
from typing import List, Dict, Any, Optional

_local_llm_instance = None


class LocalLLMService:
    """
    On-device Small Language Model (SLM) service using Qwen2.5-0.5B-Instruct.
    Runs 100% offline on standard CPU with zero cloud latency and zero data leakage.
    """

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        self.model_path = Path(
            os.path.expanduser(
                r"~/.cache/huggingface/hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/7ae557604adf67be50417f59c2c2f167def9a775"
            )
        )

    def is_available(self) -> bool:
        """Check if local model weights are present on disk."""
        if not self.model_path.exists():
            return False
        weight_file = self.model_path / "model.safetensors"
        return weight_file.exists() and weight_file.stat().st_size > 800 * 1024 * 1024

    def load_model(self) -> bool:
        """Loads the local Qwen2.5-0.5B-Instruct model into memory."""
        if self.is_loaded and self.model is not None and self.tokenizer is not None:
            return True

        if not self.is_available():
            return False

        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForCausalLM

            torch.set_num_threads(max(1, os.cpu_count() - 1 if os.cpu_count() else 4))

            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_path),
                local_files_only=True
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                str(self.model_path),
                local_files_only=True,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True
            )
            self.model.eval()
            self.is_loaded = True
            return True
        except Exception as e:
            print(f"[LocalLLMService] Error loading model: {e}")
            return False

    def generate_response(
        self,
        query: str,
        context_chunks: Optional[List[Dict[str, Any]]] = None,
        user_profile: Optional[Dict[str, Any]] = None,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 350
    ) -> Optional[Dict[str, Any]]:
        """
        Generates a context-grounded, articulate answer offline using local Qwen2.5-0.5B weights.
        """
        if not self.is_loaded:
            success = self.load_model()
            if not success:
                return None

        try:
            import torch

            # 1. Format User Profile Context
            profile_str = ""
            user_name = "User"
            if user_profile:
                user_name = user_profile.get("name") or "User"
                edu = user_profile.get("education") or "Not specified"
                skills = user_profile.get("skills") or "Not specified"
                goal = user_profile.get("career_goal") or "Not specified"
                exp = user_profile.get("experience") or "Not specified"
                profile_str = (
                    f"User Profile: Name: {user_name} | Goal: {goal} | Education: {edu} | "
                    f"Skills: {skills} | Experience: {exp}"
                )

            # 2. Format Document Context Chunks
            context_text = ""
            sources_set = set()
            if context_chunks:
                parts = []
                for c in context_chunks[:4]:
                    meta = c.get("metadata", {})
                    src = meta.get("source", "Document")
                    page = meta.get("page", 1)
                    sources_set.add(f"{src} (Page {page})")
                    parts.append(f"[Source: {src}, Page {page}]\n{c.get('text', '')[:400]}")
                context_text = "\n\n".join(parts)

            # 3. System Prompt
            system_prompt = (
                "You are KnowVox, an intelligent, empathetic, articulate AI Voice & Career Assistant. "
                "You run 100% on-device on the user's laptop with complete privacy and zero internet. "
                "Provide direct, high-value, well-structured answers using clear bullet points and headings. "
                "Always ground your answer in the provided document context or user profile if relevant."
            )
            if profile_str:
                system_prompt += f"\n\n{profile_str}"

            # 4. Assemble Messages
            messages = [{"role": "system", "content": system_prompt}]

            if conversation_history:
                for h in conversation_history[-3:]:
                    role = "user" if h.get("role") == "user" else "assistant"
                    messages.append({"role": role, "content": h.get("content", "")[:250]})

            user_content = query
            if context_text:
                user_content = f"CONTEXT FROM UPLOADED DOCUMENTS:\n{context_text}\n\nUSER QUESTION:\n{query}"

            messages.append({"role": "user", "content": user_content})

            # 5. Tokenize and Generate
            text_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            inputs = self.tokenizer([text_prompt], return_tensors="pt")

            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    pad_token_id=self.tokenizer.eos_token_id
                )

            gen_tokens = outputs[0][inputs.input_ids.shape[1]:]
            answer = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

            if not answer:
                return None

            return {
                "answer": answer,
                "sources": sorted(list(sources_set))
            }
        except Exception as e:
            print(f"[LocalLLMService] Generation error: {e}")
            return None


def get_local_llm_service() -> LocalLLMService:
    global _local_llm_instance
    if _local_llm_instance is None:
        _local_llm_instance = LocalLLMService()
    return _local_llm_instance
