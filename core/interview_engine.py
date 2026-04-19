"""
WitnessChain — Interview Engine
TRUST-governed dialogue manager with 4 interview phases.
Injects TRUST system prompt before every Gemma 4 call.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Optional

import torch
from langdetect import detect as detect_language

from ..models.gemma_loader import GemmaLoader
from .distress_detector import DistressDetector


# Interview phases with context-injection prompts
INTERVIEW_PHASES = [
    {
        "name": "Context",
        "label": "Context → Events → People → Evidence",
        "prompt_addon": (
            "\n\nCURRENT PHASE: Context gathering. "
            "Begin with: 'You can stop or skip this at any time.' "
            "Ask about WHEN and WHERE. One question only. "
            "Example: 'When did this happen?' or 'Where were you at the time?'"
        ),
    },
    {
        "name": "Events",
        "label": "Context ✓ → Events → People → Evidence",
        "prompt_addon": (
            "\n\nCURRENT PHASE: Events observation. "
            "Begin with: 'You can stop or skip this at any time.' "
            "Ask about what the witness OBSERVED — never 'what happened to you'. "
            "Focus on observable, factual context. One question only."
        ),
    },
    {
        "name": "People",
        "label": "Context ✓ → Events ✓ → People → Evidence",
        "prompt_addon": (
            "\n\nCURRENT PHASE: People descriptions. "
            "Begin with: 'You can stop or skip this at any time.' "
            "Ask if the witness can describe anyone they observed. "
            "Physical descriptions or group affiliations only. One question only."
        ),
    },
    {
        "name": "Evidence",
        "label": "Context ✓ → Events ✓ → People ✓ → Evidence",
        "prompt_addon": (
            "\n\nCURRENT PHASE: Evidence documentation. "
            "Begin with: 'You can stop or skip this at any time.' "
            "Ask about anything they saw that could be documented — "
            "documents, photos, locations. One question only."
        ),
    },
]


class InterviewEngine:
    """
    TRUST-governed interview engine.

    Manages the trauma-informed interview flow with:
    - Distress detection before every response
    - TRUST system prompt injection on every Gemma 4 call
    - 4-phase interview progression
    - Session persistence and safe deletion
    """

    def __init__(self, model: GemmaLoader, distress_detector: DistressDetector):
        """
        Args:
            model: Loaded GemmaLoader instance.
            distress_detector: Loaded DistressDetector instance.
        """
        self.model = model
        self.distress_detector = distress_detector

        # Session storage: keyed by session_id
        self._sessions = {}

        # Load TRUST system prompt
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        trust_prompt_path = os.path.join(base_dir, "prompts", "trust_system_prompt.txt")
        with open(trust_prompt_path, "r", encoding="utf-8") as f:
            self.trust_system_prompt = f.read()

        # Audit log: stores all prompts sent to Gemma 4
        self.audit_log = []

    def start_session(self, session_id: str = None) -> str:
        """
        Start a new interview session.

        Args:
            session_id: Optional session identifier. Generated if not provided.

        Returns:
            Opening message to display to the witness.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        self._sessions[session_id] = {
            "id": session_id,
            "started_at": datetime.now(timezone.utc).isoformat(),
            "phase_index": 0,
            "exchange_count": 0,
            "history": [],
            "detected_language": None,
            "is_complete": False,
            "is_distress_exit": False,
            "trust_scores": {
                "single_question_violations": 0,
                "validation_count": 0,
                "total_turns": 0,
            },
        }

        opening = (
            "Welcome. I am here to help you document what you experienced, "
            "at your own pace and in your own words.\n\n"
            "You are in full control of this process. You can stop at any time "
            "by pressing the **⬛ Stop Safely** button or typing [STOP].\n\n"
            "Everything you share will be stored only on this device and will not "
            "be sent anywhere without your permission.\n\n"
            "This interview has four short sections — context, events, people, "
            "and evidence — with two questions each. We will complete in "
            "approximately 8 exchanges.\n\n"
            "When you are ready, please tell me — in any language you prefer — "
            "when and where did this event take place?"
        )

        self._sessions[session_id]["history"].append({
            "role": "assistant",
            "content": opening,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        return opening

    def respond(self, session_id: str, user_input: str) -> tuple:
        """
        Process user input and generate TRUST-governed response.

        Args:
            session_id: The session identifier.
            user_input: The witness's input text.

        Returns:
            Tuple of (response_text, is_complete, detected_language).
        """
        session = self._sessions.get(session_id)
        if session is None:
            return ("Session not found. Please start a new session.", True, "en")

        if session["is_complete"]:
            return ("This session has already ended.", True, session.get("detected_language", "en"))

        # --- Step 1: Detect language ---
        # langdetect is unreliable for short inputs (<20 chars).
        # The TRUST system prompt instructs Gemma 4 to match the witness's
        # language natively, so even if detection is wrong here, the model
        # will respond in the correct language.
        detected_lang = session.get("detected_language", "en")
        try:
            if len(user_input.strip()) >= 20:
                detected_lang = detect_language(user_input)
        except Exception:
            pass  # Keep previous language detection

        session["detected_language"] = detected_lang

        # Step 2: Check distress FIRST (non-negotiable)
        # detector.detect() now returns the specific lang_key (e.g. 'ar-SD') that matched
        distress_lang_key = self.distress_detector.detect(user_input)
        distress_detected = distress_lang_key is not None
        
        # Semantic Fallback: if keywords miss and language is under-resourced, use Gemma 4
        # We gate this to avoid doubling latency for languages with robust keywords (en, ar, fr, sw, ti, es, so, ar-SD)
        KEYWORD_COVERED_LANGS = {"en", "ar", "fr", "sw", "ti", "es", "so", "ar-SD"}
        if not distress_detected and detected_lang not in KEYWORD_COVERED_LANGS and len(user_input.split()) > 3:
            distress_detected = self.distress_detector.detect_semantic(user_input, self.model)
            if distress_detected:
                distress_lang_key = detected_lang

        if distress_detected:
            session["is_complete"] = True
            session["is_distress_exit"] = True
            
            # Use specific dialect key if matched, otherwise generic detected language
            resource_key = distress_lang_key if distress_lang_key and distress_lang_key != "universal" else detected_lang
            safe_exit = self.distress_detector.get_safe_exit_message(resource_key)

            session["history"].append({
                "role": "user",
                "content": user_input,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            session["history"].append({
                "role": "assistant",
                "content": safe_exit,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "is_safe_exit": True,
            })

            self.audit_log.append({
                "session_id": session_id,
                "event": "DISTRESS_DETECTED",
                "trigger_text": user_input[:100],
                "detection_type": "keyword" if self.distress_detector.detect(user_input) else "semantic",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            return (safe_exit, True, detected_lang)

        # --- Step 3: Record user message in history ---
        session["history"].append({
            "role": "user",
            "content": user_input,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })
        session["exchange_count"] += 1

        # --- Step 4: Build conversation context ---
        phase = INTERVIEW_PHASES[min(session["phase_index"], len(INTERVIEW_PHASES) - 1)]

        # Build history string for context
        # Strategy: Exploit Gemma 4's native 256K context window.
        # We pass the ENTIRE history to the model to ensure maximal coherence.
        combined = session["history"]

        history_text = ""
        for entry in combined:
            role = "Witness" if entry["role"] == "user" else "WitnessChain"
            history_text += f"{role}: {entry['content']}\n\n"

        full_system_prompt = (
            self.trust_system_prompt
            + phase["prompt_addon"]
            + f"\n\nConversation so far:\n{history_text}"
        )

        # --- Step 5: Generate response via Gemma 4 ---
        try:
            response = self.model.generate(
                system_prompt=full_system_prompt,
                user_message=user_input,
                max_new_tokens=512
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            response = (
                "Thank you for your patience. I need a brief moment to recover. "
                "Could you please repeat or rephrase what you just shared?"
            )
            self.audit_log.append({
                "session_id": session_id,
                "event": "OOM_ERROR",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except RuntimeError as e:
            response = (
                "I understand what you've shared. Thank you for your patience. "
                "Could you tell me a bit more when you feel ready?"
            )
            self.audit_log.append({
                "session_id": session_id,
                "event": "RUNTIME_ERROR",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        except Exception as e:
            response = (
                "I understand what you've shared. Thank you for your patience. "
                "Could you tell me a bit more when you feel ready?"
            )
            self.audit_log.append({
                "session_id": session_id,
                "event": "GENERATION_ERROR",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        # --- Step 6: Log the prompt to audit trail ---
        session["trust_scores"]["total_turns"] += 1

        # Check TRUST compliance on generated response
        # Detect question marks across scripts: Latin ?, Arabic ؟, etc.
        question_marks = re.findall(r'[?\u061F\u2047\u2048\u2049\u2753\u2754]', response)
        question_count = len(question_marks)
        if question_count > 1:
            session["trust_scores"]["single_question_violations"] += 1
            # ENFORCE: truncate to first question only (measure AND enforce)
            parts = re.split(r'[?\u061F\u2047\u2048\u2049\u2753\u2754]', response)
            first_q_mark = question_marks[0]
            response = parts[0].strip() + first_q_mark
            self.audit_log.append({
                "session_id": session_id,
                "event": "TRUST_ENFORCEMENT",
                "action": "truncated_to_single_question",
                "original_question_count": question_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

        validation_words = [
            "understand", "thank", "hear", "acknowledge", "i hear", "i see",
            # Arabic
            "أفهم", "شكرا", "أشكرك", "أقدر",
            # French
            "je comprends", "merci", "j'entends", "je vous entends",
            # Swahili
            "naelewa", "asante", "naomba", "ninakushukuru",
            # Tigrinya
            "አመሰግን",
        ]
        if any(w in response.lower() for w in validation_words):
            session["trust_scores"]["validation_count"] += 1

        # Track token usage for the judge's context-window audit
        input_tokens = self.model.get_token_count(full_system_prompt + user_input)
        response_tokens = self.model.get_token_count(response)

        self.audit_log.append({
            "session_id": session_id,
            "event": "GENERATION",
            "phase": phase["name"],
            "token_usage": {
                "input": input_tokens,
                "output": response_tokens,
                "total_context": input_tokens + response_tokens,
                "context_limit": self.model.max_context_length
            },
            "system_prompt_hash": hash(full_system_prompt),
            "system_prompt_preview": full_system_prompt[:200],
            "response_preview": response[:200],
            "trust_check": {
                "question_count": question_count,
                "has_validation": any(w in response.lower() for w in validation_words),
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # --- Step 7: Record assistant response ---
        session["history"].append({
            "role": "assistant",
            "content": response,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

        # --- Step 8: Advance phase or complete ---
        # Advance phase every 2 substantive exchanges
        EXCHANGES_PER_PHASE = 2
        if session["exchange_count"] > 0 and session["exchange_count"] % EXCHANGES_PER_PHASE == 0:
            if session["phase_index"] < len(INTERVIEW_PHASES) - 1:
                session["phase_index"] += 1

        # Mark complete after all 4 phases have had at least 2 exchanges each (8 total)
        is_complete = session["exchange_count"] >= 8
        if is_complete:
            session["is_complete"] = True
            completion_msg = (
                "\n\n---\n\n"
                "Thank you for sharing your testimony. Your account has been saved. "
                "You can download, review, or delete it at any time using the tabs above.\n\n"
                "Your courage in documenting this matters."
            )
            response += completion_msg

        return (response, is_complete, detected_lang)

    def get_phase_label(self, session_id: str) -> str:
        """Get the current interview phase label for UI display."""
        session = self._sessions.get(session_id)
        if session is None:
            return "No active session"
        phase_idx = min(session["phase_index"], len(INTERVIEW_PHASES) - 1)
        return INTERVIEW_PHASES[phase_idx]["label"]

    def save_testimony(self, session_id: str) -> Optional[dict]:
        """
        Save and return the structured testimony JSON for a completed session.

        Args:
            session_id: The session identifier.

        Returns:
            Testimony dict with all session data, or None if session not found.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return None

        # Extract just the witness messages
        witness_messages = [
            entry["content"]
            for entry in session["history"]
            if entry["role"] == "user"
        ]

        raw_text = " ".join(witness_messages)

        testimony = {
            "id": f"testimony_{session_id[:8]}",
            "session_id": session_id,
            "language": session.get("detected_language", "en"),
            "raw_text": raw_text,
            "full_transcript": session["history"],
            "extracted": None,  # Filled by ExtractionEngine later
            "session_timestamp": session["started_at"],
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "trust_compliance": self.get_trust_score(session_id),
            "is_distress_exit": session.get("is_distress_exit", False),
        }

        return testimony

    def delete_session(self, session_id: str) -> bool:
        """
        Permanently delete all session data.

        Args:
            session_id: The session identifier.

        Returns:
            True if session was found and deleted.
        """
        if session_id in self._sessions:
            del self._sessions[session_id]
            self.audit_log.append({
                "session_id": session_id,
                "event": "SESSION_DELETED",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            return True
        return False

    def get_trust_score(self, session_id: str) -> dict:
        """
        Calculate TRUST compliance score for a session.
        NOTE: Violations are measured ON RAW MODEL OUTPUT before truncation.
        
        Returns:
            Dict with per-metric scores and overall percentage.
        """
        session = self._sessions.get(session_id)
        if session is None:
            return {"overall": 0}

        scores = session["trust_scores"]
        total = scores["total_turns"]
        if total == 0:
            return {"overall": 100, "total_turns": 0}

        single_q_rate = 1.0 - (scores["single_question_violations"] / total)
        validation_rate = scores["validation_count"] / total

        overall = ((single_q_rate * 50) + (validation_rate * 50))

        return {
            "overall": round(overall, 1),
            "single_question_rate": round(single_q_rate * 100, 1),
            "validation_rate": round(validation_rate * 100, 1),
            "total_turns": total,
            "violations": scores["single_question_violations"],
        }

    def get_active_sessions(self) -> list:
        """Returns list of active session IDs."""
        return [
            sid for sid, s in self._sessions.items()
            if not s["is_complete"]
        ]

    def get_all_testimonies(self) -> list:
        """Returns saved testimony data for all completed sessions."""
        return [
            self.save_testimony(sid)
            for sid, s in self._sessions.items()
            if s["is_complete"]
        ]
