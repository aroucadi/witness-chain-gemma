"""
WitnessChain — Cross-Reference Engine
256K context window testimony corroboration using Gemma 4.
This is the STAR TECHNICAL FEATURE of WitnessChain.
"""

import json
import os

import torch
from typing import Optional

from ..models.gemma_loader import GemmaLoader


class CrossReferenceEngine:
    """
    Cross-references multiple testimonies using Gemma 4's 256K context window.

    Packs all testimonies into a single context window with XML-tag separators,
    then runs a single inference call to surface corroborating evidence,
    timeline reconstruction, discrepancies, and evidence gaps.
    """

    MAX_CONTEXT_TOKENS = 262144  # Gemma 4 256K context window

    def __init__(self, model: GemmaLoader):
        """
        Args:
            model: Loaded GemmaLoader instance.
        """
        self.model = model

        # Load cross-reference prompt template
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(base_dir, "prompts", "crossref_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.crossref_prompt_template = f.read()

    def pack_testimonies(self, testimonies: list) -> str:
        """
        Format testimonies with XML tags for context packing.

        Each testimony is wrapped in <testimony> tags with ID, language,
        and timestamp attributes.

        Args:
            testimonies: List of testimony dicts with 'id', 'language',
                        'raw_text', and 'session_timestamp' fields.

        Returns:
            Packed testimony string ready for context injection.
        """
        packed = ""
        for t in testimonies:
            testimony_id = t.get("id", "unknown")
            lang = t.get("language", "unknown")
            ts = t.get("session_timestamp", "unknown")
            raw_text = t.get("raw_text", "")

            packed += f'\n<testimony id="{testimony_id}" language="{lang}" timestamp="{ts}">\n'
            packed += raw_text
            packed += "\n</testimony>\n"

        return packed

    def get_token_usage(self, packed: str) -> dict:
        """
        Calculate token usage statistics for the packed testimony context.

        This is surfaced in the UI as a key technical credibility signal.

        Args:
            packed: The packed testimony string.

        Returns:
            Dict with tokens_used, tokens_available, percentage, and display string.
        """
        tokens_used = self.model.get_token_count(packed)
        tokens_available = self.MAX_CONTEXT_TOKENS
        percentage = (tokens_used / tokens_available) * 100

        return {
            "tokens_used": tokens_used,
            "tokens_available": tokens_available,
            "percentage": round(percentage, 2),
            "display": f"{tokens_used:,} / {tokens_available:,} tokens ({percentage:.1f}%)",
        }

    def analyse(self, testimonies: list) -> dict:
        """
        Run cross-reference analysis on all provided testimonies.

        Packs all testimonies into Gemma 4's 256K context window and runs
        a single inference call with the cross-reference prompt.

        Args:
            testimonies: List of testimony dicts to cross-reference.

        Returns:
            Cross-reference analysis dict with corroborated_facts, timeline,
            geographic_cluster, discrepancies, and evidence_gaps.
            Includes token_usage metadata.
        """
        if not testimonies:
            return {
                "error": "No testimonies provided for cross-reference analysis.",
                "token_usage": {"tokens_used": 0, "tokens_available": self.MAX_CONTEXT_TOKENS, "percentage": 0},
            }

        # Pack testimonies into XML-tagged context
        packed = self.pack_testimonies(testimonies)

        # Build full prompt using sentinel substitution.
        # Using {{SENTINEL}} style tokens to avoid accidental collision with
        # numerical content in testimony text (e.g. dates, counts, distances).
        prompt = self.crossref_prompt_template.replace(
            "{{N_TESTIMONIES}}", str(len(testimonies))
        ).replace(
            "{{PACKED_TESTIMONIES}}", packed
        )

        # Calculate token usage BEFORE inference (for UI display)
        token_usage = self.get_token_usage(prompt)

        print(f"[WitnessChain] Cross-reference: {len(testimonies)} testimonies, "
              f"{token_usage['display']}")

        # Verify we're within context limits.
        # Subtract generation budget so input + output never exceeds model's max sequence length.
        MAX_GENERATION_TOKENS = 2048
        safe_input_limit = self.MAX_CONTEXT_TOKENS - MAX_GENERATION_TOKENS

        if token_usage["tokens_used"] > safe_input_limit * 0.95:
            return {
                "error": (
                    f"Token count ({token_usage['tokens_used']:,}) exceeds 95% of safe input limit "
                    f"({safe_input_limit:,} = {self.MAX_CONTEXT_TOKENS:,} − {MAX_GENERATION_TOKENS:,} "
                    f"generation budget). Reduce testimony count."
                ),
                "token_usage": token_usage,
            }

        # Single inference call — all corroboration surfaced at once
        try:
            response = self.model.generate_long(
                prompt=prompt,
                max_new_tokens=2048
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            return {
                "error": (
                    "GPU memory exhausted during cross-reference analysis. "
                    "Try with fewer testimonies or shorter texts."
                ),
                "token_usage": token_usage,
                "corroborated_facts": [],
                "timeline": [],
                "geographic_cluster": {},
                "discrepancies": [],
                "evidence_gaps": [],
            }
        except RuntimeError as e:
            return {
                "error": f"Cross-reference inference failed (RuntimeError): {str(e)}",
                "token_usage": token_usage,
                "corroborated_facts": [],
                "timeline": [],
                "geographic_cluster": {},
                "discrepancies": [],
                "evidence_gaps": [],
            }
        except Exception as e:
            return {
                "error": f"Cross-reference inference failed: {str(e)}",
                "token_usage": token_usage,
                "corroborated_facts": [],
                "timeline": [],
                "geographic_cluster": {},
                "discrepancies": [],
                "evidence_gaps": [],
            }

        # Parse JSON response — first attempt
        parsed = self._parse_json_response(response)
        if parsed is not None:
            parsed["token_usage"] = token_usage
            return parsed

        # Retry with stricter prompt on malformed JSON
        retry_prompt = (
            "The previous response was not valid JSON. "
            "Return ONLY a valid JSON object with these exact top-level keys: "
            "corroborated_facts, timeline, geographic_cluster, discrepancies, evidence_gaps. "
            "No markdown, no explanation, no preamble.\n\n"
            + prompt
        )
        try:
            retry_response = self.model.generate_long(
                prompt=retry_prompt,
                max_new_tokens=2048
            )
            parsed = self._parse_json_response(retry_response)
            if parsed is not None:
                parsed["token_usage"] = token_usage
                return parsed
        except Exception:
            pass

        # Final fallback: return structured error with raw response
        return {
            "error": "Failed to parse cross-reference JSON output after retry",
            "raw_response": response[:2000],
            "token_usage": token_usage,
            "corroborated_facts": [],
            "timeline": [],
            "geographic_cluster": {},
            "discrepancies": [],
            "evidence_gaps": [],
        }

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """Parse JSON from model response with fallback strategies."""
        text = response.strip()

        # Direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Extract from markdown code blocks
        if "```json" in text:
            try:
                json_str = text.split("```json")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        if "```" in text:
            try:
                json_str = text.split("```")[1].split("```")[0].strip()
                return json.loads(json_str)
            except (IndexError, json.JSONDecodeError):
                pass

        # Find JSON object boundaries
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        return None
