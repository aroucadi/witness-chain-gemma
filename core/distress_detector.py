"""
WitnessChain — Distress Signal Detector
TIC safety layer: multilingual keyword detection with crisis resource lookup.
Synchronous detection to ensure prompt exit before inference.
"""

import json
import os
import re
import threading
from typing import Optional


# Comprehensive multilingual distress keyword list
# Covers: English, Arabic, French, Swahili, Tigrinya, and universal signals
DISTRESS_KEYWORDS = {
    "en": [
        "stop", "enough", "can't", "cannot", "too much", "hurts", "scared",
        "afraid", "don't want", "leave me", "too painful", "i'm done",
        "please stop", "no more", "i can't do this", "this is too hard",
        "i need to stop", "help me", "i'm not okay", "it's too much",
        "i want to leave", "end this", "i refuse", "i'm breaking down",
        "overwhelmed", "panic", "terrified", "nightmare", "flashback"
    ],
    "ar": [
        "كفى", "أوقف", "لا أستطيع", "خائف", "خوف", "يؤلم", "توقف",
        "لا أريد", "مؤلم جدا", "انتهيت", "أرجوك توقف", "لا أكثر",
        "ساعدوني", "أنا خائفة", "لا أقدر", "هذا كثير", "أوجعني",
        "مرعوب", "كابوس", "أريد أن أتوقف"
    ],
    "ar-SD": [
        "كفى", "أوقف", "لا أستطيع", "خائف", "خوف", "توقف",
        "يا زول", "خلاص", "ما بقدر", "ساعدونا", "داير أقيف",
        "تعبتا شديد", "مرعوب", "كابوس", "أرجوك أوقف"
    ],
    "fr": [
        "arrêtez", "assez", "je ne peux pas", "trop", "j'ai peur",
        "ça fait mal", "je veux arrêter", "laissez-moi", "c'est trop",
        "je n'en peux plus", "aidez-moi", "je refuse", "stop",
        "je suis terrifiée", "cauchemar", "panique", "c'est insupportable",
        "je craque", "plus jamais", "je souffre"
    ],
    "sw": [
        "simama", "inatosha", "siwezi", "ninaogopa", "inaumiza",
        "tafadhali acha", "sitaki", "nimechoka", "nisaidie",
        "siendi", "ni mbaya sana", "naogopa sana", "acha",
        "hataki", "imenitosha", "sitaki kuendelea", "niache",
        "nahuzunika", "jinambo", "inaniuma", "nimevunjika",
        "nimeshikwa na hofu", "sitaki tena"
    ],
    "ti": [
        "አቁም", "በቃ", "አልችልም", "ፈሪሐ", "የሕምም", "ኣቁም",
        "ኣይደልን", "ሓግዙኒ", "ይኣክል", "ኣይክእልን"
    ],
    "so": [
        "jooji", "filo", "cabsi", "caawimaad", "xanuun", "ma rabo",
        "i daa", "waa igu filan", "ma rabo inaan hadlo", "ii jooji",
        "fadlan jooji", "kama bixi karo", "waxay iila muuqataa xumaan"
    ],
    "my": [
        "ရပ်", "တော်ပြီ", "ကြောက်တယ်", "ကူညီပါ", "နာတယ်",
        "မလုပ်ချင်တော့ဘူး", "ဆက်မပြောချင်ဘူး", "ရပ်ပေးပါ", "အရမ်းနာတယ်",
        "ကြောက်နေပြီ", "ကူညီကြပါ", "မခံနိုင်တော့ဘူး"
    ],
    "es": [
        "parar", "basta", "miedo", "ayuda", "duele", "no quiero",
        "deténgase", "por favor paren", "ya es suficiente", "no puedo más",
        "auxilio", "tengo miedo", "déjenme ir", "esto es demasiado",
        "ya no quiero hablar", "me duele mucho", "socorro"
    ],
    "universal": [
        "[STOP]", "[ENOUGH]", "[EXIT]", "!!!",
        "STOP", "ENOUGH", "EXIT", "HELP", "QUIT"
    ],
}


class DistressDetector:
    """
    Multilingual distress signal detector for the TIC safety layer.

    Trigger threshold: Any single keyword match triggers immediate safe exit.
    False positives (premature exits) are acceptable and recoverable.
    False negatives (missed distress) are NOT acceptable.
    """

    def __init__(self, crisis_resources_path: str = None):
        """
        Args:
            crisis_resources_path: Path to crisis_resources.json.
                                   Defaults to data/crisis_resources.json relative to package.
        """
        self.keywords = DISTRESS_KEYWORDS

        if crisis_resources_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            crisis_resources_path = os.path.join(base_dir, "data", "crisis_resources.json")

        self.crisis_resources_path = crisis_resources_path
        self._crisis_resources = None
        self._load_crisis_resources()

    def _load_crisis_resources(self):
        """Load crisis resources from JSON file."""
        try:
            with open(self.crisis_resources_path, "r", encoding="utf-8") as f:
                self._crisis_resources = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            print(f"[WitnessChain] Warning: Could not load crisis resources: {e}")
            self._crisis_resources = {
                "en": "If you are in crisis, please contact your local emergency services or a crisis helpline.",
            }

    def detect(self, text: str) -> Optional[str]:
        """
        Detect distress signals in the given text.

        Checks all language keyword sets. Returns the key of the first match.

        Args:
            text: The user's input text to scan.

        Returns:
            The language key (e.g., 'en', 'ar-SD') if detected, else None.
        """
        if not text:
            return None

        text_lower = text.lower().strip()

        # Check universal signals first (case-sensitive for bracketed commands)
        for keyword in self.keywords.get("universal", []):
            if keyword in text or keyword.lower() in text_lower:
                return "universal"

        # Check all language-specific keywords
        # Priority: Check regional dialects (like ar-SD) before generic ones (ar)
        lang_keys = sorted(self.keywords.keys(), key=lambda x: len(x), reverse=True)
        for lang in lang_keys:
            if lang == "universal":
                continue
            keywords = self.keywords[lang]
            for keyword in keywords:
                if self._keyword_match(keyword, text_lower, text):
                    return lang

        return None

    def _keyword_match(self, keyword: str, text_lower: str, text_original: str) -> bool:
        """
        Match a keyword using word-boundary rules for single words and
        substring matching for multi-word phrases.

        Single-word boundary matching reduces false positives (e.g. 'stop' won't
        match 'stopped') while maintaining zero-miss tolerance for multi-word
        distress phrases.
        """
        if ' ' in keyword:
            # Multi-word phrase: substring match is appropriate
            return keyword.lower() in text_lower
        # Single word: require word boundaries to avoid partial matches
        return bool(re.search(r'\b' + re.escape(keyword.lower()) + r'\b', text_lower))

    def detect_semantic(self, text: str, model_loader) -> bool:
        """
        Semantic Distress Detection (Fallback).
        Uses Gemma 4 to evaluate distress in high-complexity cases or
        unsupported languages where keyword matching fails.
        """
        if not text or len(text.split()) < 3:
            return False

        prompt = (
            "Evaluate if the following witness testimony snippet indicates a desire to STOP, "
            "extreme distress, a panic attack, or a refusal to continue. "
            "Respond ONLY with 'TRUE' or 'FALSE'.\n\n"
            f"Snippet: \"{text}\"\n\n"
            "Result:"
        )
        
        try:
            # Use a fast, greedy decoding for safety check
            response = model_loader.generate(
                system_prompt="You are a safety classifier.",
                user_message=prompt,
                max_new_tokens=5
            )
            return "TRUE" in response.upper()
        except Exception:
            # F9: Fail-silent safety fallback. 
            # If the semantic check fails due to inference issues, we rely on keywords.
            # We avoid the length heuristic which triggers false positives on long testimonies.
            return False

    def detect_async(self, text: str, callback=None) -> threading.Thread:
        """
        Run distress detection in a separate thread (non-blocking).

        .. deprecated::
            The synchronous `detect()` method is preferred for safety-critical
            paths. Async detection creates a race condition where inference
            could begin before distress is caught.
        """
        # F7: Hardened deprecation — prevent usage in safety-critical paths
        raise NotImplementedError(
            "detect_async is deprecated for safety reasons. Use synchronous detect() instead."
        )
        # Old implementation removed to ensure safety.

    def get_crisis_resources(self, language_code: str = "en", is_online: bool = False) -> str:
        """
        Returns crisis resources for the given language using the structured schema.
        Prioritizes phone numbers for offline-first support according to mandated order.
        """
        if self._crisis_resources is None:
            return "If you are in crisis, please contact your local emergency services."

        res = self._crisis_resources.get(language_code)
        if not res:
            # Fallback to base language if specific regional key (e.g. ar-SD) not found
            base_lang = language_code.split('-')[0]
            res = self._crisis_resources.get(base_lang)
            
        if not res:
            res = self._crisis_resources.get("en")
            
        if not isinstance(res, dict):
            return str(res)

        # Build output string in priority order: Org -> Phone -> (WhatsApp if online) -> UNHCR
        org = res.get("org", "Local Support")
        phone = res.get("phone", "Emergency Services")
        whatsapp = res.get("whatsapp")
        unhcr = res.get("unhcr")
        
        lines = [
            f"{org}",
            f"📞 PHONE (Offline): {phone}"
        ]
        
        if is_online and whatsapp:
            # Rendered as text only for security/simplicity
            lines.append(f"💬 WHATSAPP: {whatsapp}")
            
        if unhcr:
            lines.append(f"🌐 UNHCR Contact: {unhcr}")
            
        return "\n".join(lines)

    def get_safe_exit_message(self, language_code: str = "en") -> str:
        """
        Returns the complete safe exit message with crisis resources.

        Args:
            language_code: ISO 639-1 language code.

        Returns:
            Full safe exit message including thanks and crisis resources.
        """
        resources = self.get_crisis_resources(language_code)

        # Multilingual safe exit messages
        exit_messages = {
            "en": f"Thank you for sharing what you could. Your testimony has been saved. "
                  f"Here are some support resources:\n\n{resources}",
            "ar": f"شكراً لمشاركتك ما استطعت. تم حفظ شهادتك. "
                  f"إليك بعض موارد الدعم:\n\n{resources}",
            "fr": f"Merci d'avoir partagé ce que vous avez pu. Votre témoignage a été sauvegardé. "
                  f"Voici des ressources de soutien :\n\n{resources}",
            "sw": f"Asante kwa kushiriki ulichoweza. Ushuhuda wako umehifadhiwa. "
                  f"Hapa kuna rasilimali za msaada:\n\n{resources}",
            "ti": f"ስለ ዝተኻኣለካ ምትካፈልካ ኣመሰግን። ምስክርነትካ ተዓቂቡ ኣሎ። "
                  f"ናይ ሓገዝ ጸጋታት:\n\n{resources}",
        }

        return exit_messages.get(
            language_code,
            exit_messages["en"]
        )
