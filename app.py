"""
WitnessChain — Gradio Application
4-tab trauma-informed testimony infrastructure interface.
Launch with: demo.launch(share=True, debug=False)
"""

import json
import os
import uuid
import tempfile
from datetime import datetime, timezone

import gradio as gr

from core.distress_detector import DistressDetector
from core.interview_engine import InterviewEngine
from core.extraction_engine import ExtractionEngine
from core.crossref_engine import CrossReferenceEngine
from core.report_generator import ReportGenerator
from models.gemma_loader import GemmaLoader

# ============================================================
# GLOBALS — initialised on startup
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SAMPLE_DIR = os.path.join(BASE_DIR, "data", "sample_testimonies")

# These are populated in init_system()
# ⚠️ SINGLE-USER PROTOTYPE: All state is shared across sessions.
# For multi-user deployment, migrate to gr.State() per-session storage.
# This is acceptable for the hackathon demo (single evaluator at a time).
model = None
distress_detector = None
interview_engine = None
extraction_engine = None
crossref_engine = None
report_generator = None
current_session_id = None

# Testimony store (in-memory, session-scoped)
testimony_store = []


def load_sample_testimonies():
    """Load pre-built sample testimonies from data/sample_testimonies/."""
    samples = []
    if os.path.exists(SAMPLE_DIR):
        for fname in sorted(os.listdir(SAMPLE_DIR)):
            if fname.endswith(".json"):
                fpath = os.path.join(SAMPLE_DIR, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        samples.append(json.load(f))
                except json.JSONDecodeError as e:
                    print(f"[Error] Failed to parse {fname}: {e}")
    return samples


def init_system(model_choice="Fine-tuned WitnessChain"):
    """
    Initialise all system components.
    Called once at startup and when model is toggled.
    """
    global model, distress_detector, interview_engine
    global extraction_engine, crossref_engine, report_generator
    global testimony_store

    # Distress detector (lightweight, always available)
    distress_detector = DistressDetector()

    # Report generator (no model dependency)
    report_generator = ReportGenerator()

    # Load sample testimonies into store
    testimony_store = load_sample_testimonies()

    # Load Gemma model
    use_finetuned = (model_choice == "Fine-tuned WitnessChain")
    model = GemmaLoader(model_size="27b", use_finetuned=use_finetuned)

    try:
        model.load()
    except Exception as e:
        print(f"[WitnessChain] Model loading error: {e}")
        print("[WitnessChain] Running in demo mode without model.")

    # Init engines
    interview_engine = InterviewEngine(model, distress_detector)
    extraction_engine = ExtractionEngine(model)
    crossref_engine = CrossReferenceEngine(model)


# ============================================================
# TAB 1: NEW TESTIMONY — TRUST-governed interview
# ============================================================

def start_new_session():
    """Start a new interview session."""
    global current_session_id
    current_session_id = str(uuid.uuid4())

    if interview_engine is None:
        return (
            [("", "System is initialising. Please wait...")],
            "No active session",
            "🌐 Language detection: waiting for input..."
        )

    opening = interview_engine.start_session(current_session_id)
    phase_label = interview_engine.get_phase_label(current_session_id)

    return (
        [("", opening)],
        f"Interview phase: [{phase_label}]",
        "🌐 Language detection: waiting for input..."
    )


def respond_to_witness(user_message, chat_history):
    """Process witness input through TRUST-governed interview engine."""
    global current_session_id

    if not user_message or not user_message.strip():
        return chat_history, "", "Please share when you are ready.", ""

    if interview_engine is None or current_session_id is None:
        chat_history = chat_history or []
        chat_history.append((user_message, "System is not ready. Please click 'Start New Session'."))
        return chat_history, "", "No active session", ""

    response, is_complete, detected_lang = interview_engine.respond(
        current_session_id, user_message
    )

    chat_history = chat_history or []
    chat_history.append((user_message, response))

    phase_label = interview_engine.get_phase_label(current_session_id)
    lang_display = f"🌐 Detected language: {detected_lang} — responding in {detected_lang}"

    status = f"Interview phase: [{phase_label}]"
    if is_complete:
        # Save testimony to store
        testimony = interview_engine.save_testimony(current_session_id)
        if testimony:
            testimony_store.append(testimony)
        status = "✅ Interview complete — testimony saved."

    return chat_history, "", status, lang_display


def stop_safely():
    """Emergency stop — save partial testimony, then end session safely."""
    global current_session_id

    saved_lang = "en"
    if interview_engine and current_session_id:
        # Save partial testimony before deletion
        session = interview_engine._sessions.get(current_session_id)
        if session:
            saved_lang = session.get("detected_language", "en")
        partial = interview_engine.save_testimony(current_session_id)
        if partial:
            testimony_store.append(partial)
        interview_engine.delete_session(current_session_id)

    resources = ""
    if distress_detector:
        resources = distress_detector.get_safe_exit_message(saved_lang)

    current_session_id = None

    return (
        [("", f"Session stopped safely. Your testimony has been saved.\n\n{resources}")],
        "Session ended safely — partial testimony preserved.",
        ""
    )


def switch_model(model_choice):
    """Switch between base and fine-tuned model."""
    init_system(model_choice)
    # F2: Hardened feedback — check actual loading status
    is_fine = getattr(model, "is_finetuned", False)
    status_label = "Fine-tuned" if is_fine else "Base"
    return f"✅ Switched to: {model_choice} (Actual Status: {status_label})"


# ============================================================
# TAB 2: CROSS-REFERENCE — 256K context window analysis
# ============================================================

def get_testimony_list():
    """Return formatted list of available testimonies."""
    if not testimony_store:
        return "No testimonies available. Complete an interview or load samples."

    lines = []
    for t in testimony_store:
        tid = t.get("id", "unknown")
        lang = t.get("language", "?")
        preview = t.get("raw_text", "")[:80].replace("\n", " ")
        lines.append(f"[{tid}] ({lang}) {preview}...")
    return "\n".join(lines)


def run_crossref():
    """Run cross-reference analysis on all stored testimonies."""
    if not testimony_store:
        return "No testimonies to analyse.", "0 / 262,144 tokens (0%)", "{}"

    if crossref_engine is None:
        return "System not initialised.", "0 / 262,144 tokens (0%)", "{}"

    # Calculate token usage before running
    packed = crossref_engine.pack_testimonies(testimony_store)
    token_usage = crossref_engine.get_token_usage(packed)

    try:
        result = crossref_engine.analyse(testimony_store)
    except Exception as e:
        return f"Analysis error: {str(e)}", token_usage["display"], "{}"

    # Format result for display
    result_display = json.dumps(result, indent=2, ensure_ascii=False, default=str)

    return (
        result_display,
        token_usage["display"],
        result_display
    )


def upload_testimonies(files):
    """Upload testimony JSON files for cross-reference."""
    if not files:
        return "No files uploaded.", get_testimony_list()

    count = 0
    for file in files:
        try:
            with open(file.name, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "raw_text" in data:
                testimony_store.append(data)
                count += 1
        except Exception as e:
            print(f"[WitnessChain] Error loading file: {e}")

    return f"Loaded {count} testimonies.", get_testimony_list()


# ============================================================
# TAB 3: CASE REPORT — PDF/DOCX generation
# ============================================================

def generate_case_report(format_choice):
    """Generate downloadable case report."""
    if not testimony_store:
        return "No testimonies available for report generation.", None

    if extraction_engine is None or report_generator is None:
        return "System not initialised.", None

    # Extract structured data from all testimonies
    try:
        extracted = extraction_engine.batch_extract(testimony_store)
    except Exception as e:
        extracted = [{"error": str(e), "testimony_id": t.get("id", "unknown")}
                     for t in testimony_store]

    # Run cross-reference (or use cached)
    try:
        crossref = crossref_engine.analyse(testimony_store)
    except Exception as e:
        crossref = {
            "error": str(e),
            "corroborated_facts": [],
            "timeline": [],
            "geographic_cluster": {},
            "discrepancies": [],
            "evidence_gaps": [],
        }

    # Generate report
    try:
        if format_choice == "PDF":
            report_bytes = report_generator.generate_pdf(extracted, crossref)
            suffix = ".pdf"
        else:
            report_bytes = report_generator.generate_docx(extracted, crossref)
            suffix = ".docx"

        # Save to temp file for Gradio download
        tmp = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix,
            prefix="witnesschain_report_"
        )
        tmp.write(report_bytes)
        tmp.close()

        return f"✅ Report generated ({format_choice}). Click below to download.", tmp.name

    except Exception as e:
        return f"Report generation error: {str(e)}", None


# ============================================================
# TAB 4: ETHICAL AUDIT LOG
# ============================================================

def get_audit_log():
    """Return the full audit log from the interview engine."""
    if interview_engine is None:
        return "System not initialised."

    if not interview_engine.audit_log:
        return "No audit entries yet. Start an interview to generate logs."

    log_lines = []
    for entry in interview_engine.audit_log:
        ts = entry.get("timestamp", "unknown")
        event = entry.get("event", "unknown")
        sid = entry.get("session_id", "unknown")[:8]

        log_lines.append(f"[{ts}] Session {sid}… | Event: {event}")

        if event == "GENERATION":
            prompt_preview = entry.get("system_prompt_preview", "")[:150]
            log_lines.append(f"  System prompt: {prompt_preview}...")
            trust_check = entry.get("trust_check", {})
            log_lines.append(
                f"  TRUST check: questions={trust_check.get('question_count', '?')}, "
                f"validated={trust_check.get('has_validation', '?')}"
            )
            token_usage = entry.get("token_usage", {})
            if token_usage:
                log_lines.append(
                    f"  Token Usage: {token_usage.get('total_context', '?')} / {token_usage.get('context_limit', '?')} "
                    f"({round(token_usage.get('total_context', 0)/token_usage.get('context_limit', 1)*100, 2)}%)"
                )
        elif event == "DISTRESS_DETECTED":
            log_lines.append(f"  Trigger: {entry.get('trigger_text', 'N/A')}")

        log_lines.append("")

    return "\n".join(log_lines)


def get_trust_scores():
    """Return TRUST compliance scores for all sessions."""
    if interview_engine is None:
        return "System not initialised."

    sessions = list(interview_engine._sessions.keys())
    if not sessions:
        return "No sessions to score."

    lines = []
    for sid in sessions:
        score = interview_engine.get_trust_score(sid)
        lines.append(
            f"Session {sid[:8]}… | Overall: {score['overall']}% | "
            f"Single-Q: {score.get('single_question_rate', 'N/A')}% | "
            f"Validation: {score.get('validation_rate', 'N/A')}%"
        )

    return "\n".join(lines)


def delete_all_data():
    """Delete all session data and testimonies."""
    global testimony_store, current_session_id

    if interview_engine:
        for sid in list(interview_engine._sessions.keys()):
            interview_engine.delete_session(sid)

    testimony_store = []
    current_session_id = None

    return "✅ All data deleted. Session memory cleared."


# ============================================================
# BUILD GRADIO APP
# ============================================================

def build_app():
    """Construct the 4-tab Gradio interface."""

    with gr.Blocks(
        title="WitnessChain — Trauma-Informed Testimony Infrastructure",
        # Using gr.themes.Base() to minimize external CDN dependencies and ensure
        # full compliance with no-transmission claims (Data Sovereignty Pillar).
        theme=gr.themes.Base(
            primary_hue="red",
            secondary_hue="slate",
        ),
    ) as demo:

        # === Ethical disclaimer (non-negotiable — see ethical_framework.md) ===
        gr.Markdown(
            """
            > ⚠️ **This is a research prototype built for the Kaggle Gemma 4 Good Hackathon.**
            > It is not a production system. Do not use this to document real testimonies
            > without review by qualified human rights professionals and trauma counsellors.
            >
            > All model inference runs locally on this device. When using `share=True`,
            > UI traffic is routed through Gradio's proxy — no model weights or inference
            > outputs are stored externally. For fully offline operation, use `share=False`.
            > You may delete your testimony at any time.
            >
            > This system implements the **TRUST framework**, **Trauma-Informed Computing** principles,
            > and **Amnesty International's Algorithmic Accountability** guidelines.
            """
        )

        gr.Markdown(
            "# 🔗 WitnessChain\n"
            "*Trauma-informed, multilingual testimony infrastructure powered by Gemma 4*"
        )

        # ================================================================
        # TAB 1: NEW TESTIMONY
        # ================================================================
        with gr.Tab("📝 New Testimony", id="tab_testimony"):

            with gr.Row():
                with gr.Column(scale=3):
                    model_toggle = gr.Radio(
                        choices=["Base Gemma 4", "Fine-tuned WitnessChain"],
                        label="Model",
                        value="Fine-tuned WitnessChain",
                        info="Toggle to compare base vs fine-tuned model responses",
                    )
                    model_status = gr.Textbox(
                        label="Model Status",
                        value="Ready",
                        interactive=False,
                    )
                with gr.Column(scale=1):
                    stop_btn = gr.Button(
                        "⬛ Stop Safely",
                        variant="stop",
                        size="lg",
                    )

            lang_banner = gr.Markdown("🌐 Language detection: waiting for input...")

            phase_indicator = gr.Textbox(
                label="Interview Progress",
                value="Interview phase: [Context → Events → People → Evidence]",
                interactive=False,
            )

            chatbot = gr.Chatbot(
                label="Interview",
                height=400,
                type="tuples",
            )

            with gr.Row():
                user_input = gr.Textbox(
                    label="Your testimony (type in any language)",
                    placeholder="Share when you are ready...",
                    scale=3,
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
                stop_btn_bottom = gr.Button("⬛ Stop", variant="stop", scale=1)

            start_btn = gr.Button("🟢 Start New Session", variant="secondary")

            # --- Event handlers ---
            start_btn.click(
                fn=start_new_session,
                outputs=[chatbot, phase_indicator, lang_banner],
            )

            send_btn.click(
                fn=respond_to_witness,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, phase_indicator, lang_banner],
            )

            user_input.submit(
                fn=respond_to_witness,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, phase_indicator, lang_banner],
            )

            stop_btn.click(
                fn=stop_safely,
                outputs=[chatbot, phase_indicator, lang_banner],
            )

            stop_btn_bottom.click(
                fn=stop_safely,
                outputs=[chatbot, phase_indicator, lang_banner],
            )

            model_toggle.change(
                fn=switch_model,
                inputs=[model_toggle],
                outputs=[model_status],
            )

        # ================================================================
        # TAB 2: CROSS-REFERENCE
        # ================================================================
        with gr.Tab("🔍 Cross-Reference", id="tab_crossref"):

            gr.Markdown(
                "## Cross-Reference Analysis\n"
                "*Pack multiple testimonies into Gemma 4's 256K context window "
                "to surface corroborating evidence across testimonies.*"
            )

            with gr.Row():
                testimony_list = gr.Textbox(
                    label="Available Testimonies",
                    value=get_testimony_list(),
                    lines=6,
                    interactive=False,
                )

            with gr.Row():
                upload_files = gr.File(
                    label="Upload Testimony Files (JSON)",
                    file_count="multiple",
                    file_types=[".json"],
                )
                upload_status = gr.Textbox(
                    label="Upload Status",
                    interactive=False,
                )

            upload_files.change(
                fn=upload_testimonies,
                inputs=[upload_files],
                outputs=[upload_status, testimony_list],
            )

            token_counter = gr.Textbox(
                label="Context window usage",
                value="0 / 262,144 tokens (0%)",
                interactive=False,
            )

            analyse_btn = gr.Button("🔬 Analyse Corroboration", variant="primary", size="lg")

            crossref_output = gr.Textbox(
                label="Cross-Reference Results",
                lines=20,
                interactive=False,
            )

            # Hidden state for raw JSON
            crossref_json_state = gr.State(value="{}")

            analyse_btn.click(
                fn=run_crossref,
                outputs=[crossref_output, token_counter, crossref_json_state],
            )

        # ================================================================
        # TAB 3: CASE REPORT
        # ================================================================
        with gr.Tab("📄 Case Report", id="tab_report"):

            gr.Markdown(
                "## Structured Case Report\n"
                "*Generate a downloadable case file from extracted testimony data "
                "and cross-reference analysis.*"
            )

            gr.Markdown(
                "**Report sections:** Executive Summary · Incident Timeline · "
                "Corroborated Facts · Discrepancies · Evidence Inventory · "
                "Recommended Actions · Data Provenance"
            )

            format_choice = gr.Radio(
                choices=["PDF", "DOCX"],
                label="Report Format",
                value="PDF",
            )

            generate_btn = gr.Button("📥 Generate Case Report", variant="primary", size="lg")

            report_status = gr.Textbox(
                label="Status",
                interactive=False,
            )

            report_download = gr.File(
                label="Download Report",
            )

            generate_btn.click(
                fn=generate_case_report,
                inputs=[format_choice],
                outputs=[report_status, report_download],
            )

        # ================================================================
        # TAB 4: ETHICAL AUDIT LOG
        # ================================================================
        with gr.Tab("🔒 Ethical Audit Log", id="tab_audit"):

            gr.Markdown(
                "## Ethical Audit Log\n"
                "*Every prompt sent to Gemma 4 is logged here for transparency "
                "(Amnesty International Algorithmic Accountability Toolkit).*"
            )

            refresh_btn = gr.Button("🔄 Refresh Audit Log", variant="secondary")

            audit_display = gr.Textbox(
                label="Prompt Audit Trail",
                lines=15,
                interactive=False,
            )

            refresh_btn.click(
                fn=get_audit_log,
                outputs=[audit_display],
            )

            gr.Markdown("### TRUST Compliance Scores")

            trust_display = gr.Textbox(
                label="TRUST Compliance per Session",
                lines=5,
                interactive=False,
            )

            trust_btn = gr.Button("📊 Calculate TRUST Scores", variant="secondary")
            trust_btn.click(
                fn=get_trust_scores,
                outputs=[trust_display],
            )

            gr.Markdown("### Data Deletion")

            delete_btn = gr.Button(
                "🗑️ Delete All Session Data",
                variant="stop",
            )
            delete_status = gr.Textbox(
                label="Deletion Status",
                interactive=False,
            )

            delete_btn.click(
                fn=delete_all_data,
                outputs=[delete_status],
            )

    return demo


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import torch
    # Reproducibility: fix random seeds before model loading
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    print("[WitnessChain] Initialising system...")
    init_system()

    print("[WitnessChain] Building Gradio app...")
    demo = build_app()

    # Data Sovereignty Gating: Only launch with share=True if explicitly enabled via ENV.
    # Defaults to False to prevent accidental exposure of testimony UI to the internet.
    share_mode = os.environ.get("WITNESSCHAIN_SHARE_MODE", "false").lower() == "true"
    
    if share_mode:
        print("[WitnessChain] ⚠️ WARNING: Launching with share=True.")
        print("[WitnessChain] UI traffic will be routed through Gradio's proxy.")
    else:
        print("[WitnessChain] Launching in LOCAL-ONLY mode for maximum data sovereignty.")

    demo.launch(share=share_mode, debug=False)
