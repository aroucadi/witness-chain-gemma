"""
WitnessChain — Unsloth LoRA Adapter Loader
Loads fine-tuned LoRA adapter with fallback to base model.
"""

import os


class UnslothAdapter:
    """Loads the Unsloth-trained LoRA adapter for WitnessChain fine-tuned Gemma 4."""

    def __init__(self, adapter_path: str, base_model_id: str, hf_token: str):
        """
        Args:
            adapter_path: Path to saved LoRA adapter (local or HuggingFace Hub ID).
            base_model_id: Base model ID to fall back to if adapter not found.
            hf_token: HuggingFace token for model access.
        """
        self.adapter_path = adapter_path
        self.base_model_id = base_model_id
        self.hf_token = hf_token
        self.model = None
        self.tokenizer = None
        self._is_finetuned = False

    def load(self):
        """
        Load fine-tuned model. Falls back to base model if adapter not found.

        Returns:
            Tuple of (model, tokenizer).
        """
        from unsloth import FastLanguageModel

        try:
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.adapter_path,
                max_seq_length=262144,  # Gemma 4 native 256K context
                load_in_4bit=True,
                token=self.hf_token,
            )
            FastLanguageModel.for_inference(self.model)
            self._is_finetuned = True
            # Verify context length was honoured
            actual_max = getattr(self.model.config, 'max_position_embeddings', None)
            if actual_max and actual_max < 262144:
                print(f"[WitnessChain] WARNING: Requested 256K context but model config reports {actual_max}")
            print(f"[WitnessChain] Fine-tuned model loaded from {self.adapter_path}")
        except Exception as e:
            print(f"[WitnessChain] Adapter not found ({e}). Loading base model.")
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.base_model_id,
                max_seq_length=262144,  # Gemma 4 native 256K context
                load_in_4bit=True,
                token=self.hf_token,
            )
            FastLanguageModel.for_inference(self.model)
            self._is_finetuned = False

        return self.model, self.tokenizer

    def is_finetuned_available(self) -> bool:
        """Check if the fine-tuned adapter is available locally."""
        return os.path.exists(self.adapter_path)

    @property
    def is_finetuned_loaded(self) -> bool:
        """Returns True if the currently loaded model is the fine-tuned version."""
        return self._is_finetuned
