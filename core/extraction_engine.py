"""
WitnessChain — Extraction Engine
Structured entity extraction from testimony text using Gemma 4.
"""

import json
import os
from typing import Optional

from ..models.gemma_loader import GemmaLoader


class ExtractionEngine:
    """
    Extracts structured data from raw testimony text via Gemma 4 inference.

    Uses the extraction prompt template to generate JSON-structured entity
    extraction from free-text testimony.
    """

    def __init__(self, model: GemmaLoader):
        """
        Args:
            model: Loaded GemmaLoader instance.
        """
        self.model = model

        # Load extraction prompt template
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        prompt_path = os.path.join(base_dir, "prompts", "extraction_prompt.txt")
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.extraction_prompt_template = f.read()

    def extract(self, testimony_text: str) -> dict:
        """
        Extract structured entities from a single testimony.

        Args:
            testimony_text: Raw testimony text to extract from.

        Returns:
            Parsed JSON dict with extracted fields, or error dict if parsing fails.
        """
        # Format the prompt with the testimony text
        prompt = self.extraction_prompt_template.replace("{testimony_text}", testimony_text)

        # First attempt
        try:
            response = self.model.generate(
                system_prompt="You are a structured data extraction assistant. Return ONLY valid JSON.",
                user_message=prompt,
                max_new_tokens=1024
            )
        except Exception as e:
            # Return structured fallback — never crash the Gradio UI with a raw traceback
            import torch
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                error_msg = "GPU memory error during extraction. Try with a shorter testimony."
            else:
                error_msg = f"Extraction model call failed: {str(e)}"
            return {
                "error": error_msg,
                "incident_date": None, "incident_location": None,
                "incident_type": None, "perpetrator_description": None,
                "victim_count": None, "witness_relationship_to_event": None,
                "evidence_mentioned": [], "corroborating_witnesses_mentioned": False,
                "geographic_coordinates_mentioned": False, "testimony_language": None,
            }

        # Attempt to parse JSON
        parsed = self._parse_json_response(response)
        if parsed is not None:
            return parsed

        # Retry with stricter prompt on malformed JSON
        strict_prompt = (
            "The previous response was not valid JSON. "
            "Please return ONLY a valid JSON object with these exact fields: "
            "incident_date, incident_location, incident_type, perpetrator_description, "
            "victim_count, witness_relationship_to_event, evidence_mentioned, "
            "corroborating_witnesses_mentioned, geographic_coordinates_mentioned, "
            "testimony_language. Use null for unknown fields.\n\n"
            f"TESTIMONY TEXT:\n{testimony_text}"
        )

        try:
            retry_response = self.model.generate(
                system_prompt="Return ONLY a valid JSON object. No markdown, no explanation.",
                user_message=strict_prompt,
                max_new_tokens=1024
            )
        except Exception:
            retry_response = ""

        parsed = self._parse_json_response(retry_response)
        if parsed is not None:
            return parsed

        # Fallback: return error structure
        return {
            "error": "Failed to extract structured data after retry",
            "raw_response": response[:500],
            "incident_date": None,
            "incident_location": None,
            "incident_type": None,
            "perpetrator_description": None,
            "victim_count": None,
            "witness_relationship_to_event": None,
            "evidence_mentioned": [],
            "corroborating_witnesses_mentioned": False,
            "geographic_coordinates_mentioned": False,
            "testimony_language": None,
        }

    def batch_extract(self, testimonies: list) -> list:
        """
        Process multiple testimonies and extract structured data from each.

        Args:
            testimonies: List of testimony dicts, each with a 'raw_text' field.

        Returns:
            List of extracted data dicts (same order as input).
        """
        results = []
        for i, testimony in enumerate(testimonies):
            raw_text = testimony.get("raw_text", "")
            if not raw_text:
                results.append({
                    "error": "Empty testimony text",
                    "testimony_id": testimony.get("id", f"unknown_{i}"),
                })
                continue

            print(f"[WitnessChain] Extracting testimony {i+1}/{len(testimonies)}: "
                  f"{testimony.get('id', 'unknown')}")

            extracted = self.extract(raw_text)
            extracted["testimony_id"] = testimony.get("id", f"testimony_{i}")
            results.append(extracted)

        return results

    def _parse_json_response(self, response: str) -> Optional[dict]:
        """
        Attempt to parse JSON from model response, handling common formatting issues.

        Args:
            response: Raw model response text.

        Returns:
            Parsed dict or None if parsing fails.
        """
        text = response.strip()

        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from markdown code blocks
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

        # Try finding JSON object boundaries
        try:
            start = text.index("{")
            end = text.rindex("}") + 1
            json_str = text[start:end]
            return json.loads(json_str)
        except (ValueError, json.JSONDecodeError):
            pass

        return None
