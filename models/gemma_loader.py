"""
WitnessChain — Gemma 4 Model Loader
Colab-optimised 4-bit quantised loader with VRAM auto-detection.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig


class GemmaLoader:
    """Loads Gemma 4 with BitsAndBytes 4-bit quantisation for Colab environments."""

    MODEL_27B = "google/gemma-4-27b-it"
    MODEL_12B = "google/gemma-4-12b-it"
    VRAM_THRESHOLD_GB = 35  # Below this, fall back to 12B

    def __init__(self, model_size="27b", use_finetuned=False, hf_token=None):
        """
        Args:
            model_size: '27b' or '12b'. If '27b' and insufficient VRAM, auto-falls back to '12b'.
            use_finetuned: If True, load the fine-tuned LoRA adapter instead of base model.
            hf_token: HuggingFace token for model access.
        """
        self.model_size = model_size
        self.use_finetuned = use_finetuned
        self.hf_token = hf_token or os.environ.get("HF_TOKEN")
        self.model = None
        self.tokenizer = None
        self._model_id = None

    def _get_available_vram_gb(self):
        """Returns available GPU VRAM in GB. Returns 0 if no GPU."""
        if not torch.cuda.is_available():
            return 0
        try:
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            return vram_bytes / (1024 ** 3)
        except Exception:
            return 0

    def _select_model_id(self):
        """Auto-select model based on available VRAM."""
        if self.model_size == "12b":
            return self.MODEL_12B

        vram = self._get_available_vram_gb()
        if vram < self.VRAM_THRESHOLD_GB:
            print(f"[WitnessChain] Available VRAM: {vram:.1f}GB < {self.VRAM_THRESHOLD_GB}GB threshold.")
            print(f"[WitnessChain] Falling back to 12B model.")
            return self.MODEL_12B

        print(f"[WitnessChain] Available VRAM: {vram:.1f}GB — loading 27B model.")
        return self.MODEL_27B

    def load(self, offline_mode: bool = None):
        """Load model with BitsAndBytes 4-bit quantisation."""
        if offline_mode is None:
            offline_mode = os.environ.get("WITNESSCHAIN_OFFLINE", "").lower() == "true"
        self._model_id = self._select_model_id()

        if self.use_finetuned:
            from .unsloth_adapter import UnslothAdapter
            adapter_path = "models/witnesschain-lora-adapter"
            print(f"[WitnessChain] Loading fine-tuned model via UnslothAdapter...")
            adapter = UnslothAdapter(adapter_path=adapter_path, base_model_id=self._model_id, hf_token=self.hf_token)
            self.model, self.tokenizer = adapter.load()
            return self.model, self.tokenizer

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )

        # Build shared kwargs — supports offline mode for air-gapped deployments
        load_kwargs = {
            "token": self.hf_token,
            "trust_remote_code": True,
        }
        if offline_mode:
            load_kwargs["local_files_only"] = True
            print("[WitnessChain] Offline mode enabled — using cached files only.")

        print(f"[WitnessChain] Loading tokenizer: {self._model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_id,
            **load_kwargs
        )

        print(f"[WitnessChain] Loading model with 4-bit quantisation: {self._model_id}")
        self.model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            quantization_config=bnb_config,
            device_map="auto",
            **load_kwargs
        )

        print(f"[WitnessChain] Model loaded successfully: {self._model_id}")
        return self.model, self.tokenizer

    def generate(self, system_prompt: str, user_message: str, max_new_tokens=512) -> str:
        """
        Single inference call with system prompt injection.

        Args:
            system_prompt: The TRUST framework system prompt (always injected first).
            user_message: The user's message.
            max_new_tokens: Maximum tokens to generate.

        Returns:
            Generated response text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Format as chat messages for Gemma 4 instruction-tuned model.
        # Use the native system role so TRUST constraints carry full instruction weight.
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,       # Varied dialogue for natural interview flow
                temperature=0.7,
                top_p=0.9,
                top_k=50,
            )

        # Decode only the new tokens (skip input)
        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def generate_long(self, prompt: str, max_new_tokens=2048) -> str:
        """
        Long-form inference for cross-reference analysis (larger output budget).

        Args:
            prompt: The full prompt including system instructions and packed testimonies.
            max_new_tokens: Maximum tokens to generate (default 2048 for cross-ref output).

        Returns:
            Generated response text.
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        messages = [
            {"role": "user", "content": prompt}
        ]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt",
            add_generation_prompt=True
        ).to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding — deterministic JSON output
                # temperature not set: irrelevant under greedy (do_sample=False).
                # Design choice: cross-reference requires reproducible JSON structure,
                # so greedy decoding is preferred over sampling.
            )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:],
            skip_special_tokens=True
        )
        return response.strip()

    def get_token_count(self, text: str) -> int:
        """
        Returns the token count for a given text string.

        Args:
            text: Input text to tokenize.

        Returns:
            Number of tokens.
        """
        if self.tokenizer is None:
            # Fallback estimate — tokenizer not loaded.
            # This should never trigger in normal operation.
            import warnings
            warnings.warn(
                "Token count is estimated (tokenizer not loaded). "
                "Value is approximate: ~1 token per 4 characters."
            )
            return max(1, len(text) // 4)

        return len(self.tokenizer.encode(text, add_special_tokens=False))

    @property
    def model_id(self):
        """Returns the currently loaded model ID."""
        return self._model_id

    @property
    def max_context_length(self):
        """Returns the maximum context length (256K for Gemma 4)."""
        return 262144
