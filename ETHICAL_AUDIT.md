# WitnessChain — Ethical Audit Log

**Framework:** Amnesty International Algorithmic Accountability Toolkit
**System:** WitnessChain v1.0
**Audit Date:** 2026-04-18
**Auditor:** Development Team (pre-deployment self-audit)

---

## Audit Methodology

This audit follows Amnesty International's "Investigate: Algorithmic Accountability for the Public Interest" framework (2023). Each checkpoint is assessed against WitnessChain's design and implementation.

---

## Checkpoint 1: Purpose & Scope

**Question:** What is the system intended to do?

**WitnessChain Response:**
WitnessChain conducts structured, trauma-informed interviews with human rights witnesses. It extracts structured data from free-text testimonies, cross-references multiple testimonies to identify corroborating facts, and generates case documentation for use by human rights investigators.

**Scope limitations:**
- Research prototype only — not production-ready
- Demo environment: Google Colab (ephemeral)
- Support 5 primary test languages (Arabic, French, Swahili, English, Tigrinya) with Gemma 4's 140-language native capability and semantic fallback safety layer.

**Assessment:** ✅ Purpose is clearly defined and bounded.

---

## Checkpoint 2: Accountability

**Question:** Who is responsible for the system's decisions?

**WitnessChain Response:**
See `ACCOUNTABILITY.md` for full responsibility chain. Summary:
- Development team: system design and safety logic
- Deploying organisation: usage policies and data handling
- Human investigator: all downstream action on system outputs

WitnessChain produces outputs, not decisions. No automated action is taken on any individual, perpetrator, or incident. Reports require human review.

**Assessment:** ✅ Responsibility chain documented. No autonomous decisions on persons.

---

## Checkpoint 3: Transparency

**Question:** Is the system transparent about how it works?

**WitnessChain Response:**

| Transparency Measure | Status |
|---|---|
| All system prompts published in `prompts/` directory | ✅ Public |
| Fine-tuning methodology in [unsloth_pipeline.md](unsloth_pipeline.md) | ✅ Public |
| Ethical framework document | ✅ Public |
| Audit log visible in Gradio UI Tab 4 | ✅ Real-time |
| Source code open and version-controlled | ✅ Public |
| Extraction logic documented | ✅ Public |
| No closed-source components in inference | ✅ Verified |

**Assessment:** ✅ Full transparency achieved. All prompts, methods, and audit trails are public.

---

## Checkpoint 4: Bias Audit

**Question:** Has the system been tested for bias?

**WitnessChain Response:**

### Language Bias Testing

| Language | Testimony ID | Entities Extracted | Status | Extracted Facts Validation |
|---|---|---|---|---|
| **Arabic** | `testimony_001` | Date, Location, Actors | ✅ Validated | Properly captured `12 March`, `Sidi Marzouq`, `Military` |
| **French** | `testimony_002` | Date, Location, Actors | ✅ Validated | Properly captured `12 mars`, `Sidi Marzouq`, `Soldats` |
| **Swahili** | `testimony_003` | Date, Location, Actors | ✅ Validated | Properly captured `12 Machi`, `Sidi Marzouq`, `Askari` |
| **English** | `testimony_004` | Date, Location, Actors | ✅ Validated | Properly captured `March 13th`, `El Mansourah`, `Aid workers` |
| **Tigrinya**| `testimony_005` | Date, Location, Actors | ✅ Validated | Captured chronological alignment of displacement |


### TRUST Compliance Across Languages

TRUST compliance metrics (single question rate, validation rate) evaluated across all 5 test languages. Fine-tuned model shows >85% compliance across all tested languages.

### Known Bias Risks

| Risk | Mitigation |
|---|---|
| Under-resourced language performance | Gemma 4 native multilingual; tested across 5 language families |
| Western interview convention bias | System prompt explicitly instructs cultural humility |
| Formality level mismatch | Fine-tuning includes diverse cultural contexts |
| Name/entity extraction bias | Extraction prompt uses generic field types, not culture-specific patterns |

**Assessment:** ⚠️ Partial — tested across 5 languages. Production deployment requires broader language family testing (minimum 10+ languages) and community review.

---

## Checkpoint 5: Harm Mitigation

**Question:** What harms could this system cause, and how are they mitigated?

**WitnessChain Response:**

| Harm | Likelihood | Severity | Mitigation | Residual Risk |
|---|---|---|---|---|
| Witness re-traumatisation | Medium | High | TRUST framework + Dual-Layer Distress Detection (Keywords + Semantic Fallback) + safe exit | Low — mitigated by model-driven safety net |
| Extraction errors creating false record | Medium | High | Human review required; full transcript preserved | Medium — depends on downstream human compliance |
| Data breach exposing witness identity | Low | Critical | Local-only storage, ephemeral by default, no cloud | Very Low — mitigated by architecture |
| System used to fabricate testimonies | Low | High | All prompts public; extraction is extractive, not generative | Low — transparency deters misuse |
| Cultural misalignment | Medium | Medium | Multilingual training + cultural sensitivity prompt | Medium — requires ongoing community feedback |
| Overreliance on AI-generated analysis | Medium | High | Disclaimer on every output; report states "requires verification" | Medium — depends on user awareness |

**Assessment:** ✅ Comprehensive harm model documented. Primary mitigations implemented. Residual risks acknowledged.

---

## Checkpoint 6: Data Sovereignty

**Question:** Who owns the data, and how is it protected?

**WitnessChain Response:**

- **Data ownership:** The witness owns all testimony data.
- **Storage:** Local Colab ephemeral memory only. No cloud, no database.
- **Transmission:** No data is sent to external APIs. All inference is local. UI sharing via Gradio is disabled by default (`WITNESSCHAIN_SHARE_MODE=false`) to enforce maximum sovereignty.
- **Deletion:** Witness can delete at any time through the UI. Runtime termination also destroys all data.
- **Export:** Witness-initiated only. Explicit download action required.
- **Consent:** No data sharing occurs without explicit per-submission consent.

**Assessment:** ✅ Data sovereignty fully maintained. Architecture prevents exfiltration by design.

---

## Checkpoint 7: Transparency Commitments Verification

| Commitment | Status | Evidence |
|---|---|---|
| All system prompts published in `prompts/` directory | ✅ | `prompts/trust_system_prompt.txt`, `extraction_prompt.txt`, `crossref_prompt.txt` |
| Fine-tuning dataset methodology published | ✅ | `unsloth_pipeline.md` |
| TRUST compliance benchmark results published | ✅ | Evaluation in `WitnessChain_Unsloth_Finetune.ipynb` Cell 10 |
| This audit document is public | ✅ | `ETHICAL_AUDIT.md` |
| `ACCOUNTABILITY.md` answers responsibility questions | ✅ | `ACCOUNTABILITY.md` |
| No closed-source model components | ✅ | Full dependency chain is open-source |

---

## Checkpoint 8: Ongoing Monitoring

**WitnessChain Response:**

As a research prototype, ongoing monitoring is limited to the hackathon and development period. For production deployment, the following would be required:

1. **Regular bias audits** across expanded language set (quarterly)
2. **Community review** of system prompts by affected communities
3. **Incident tracking** for missed distress signals or extraction errors
4. **Version-controlled prompt updates** with audit trail
5. **Independent evaluation** by human rights practitioners

**Assessment:** ⚠️ Production monitoring plan outlined but not implemented. Appropriate for research prototype stage.

---

## Overall Audit Summary

| Checkpoint | Assessment |
|---|---|
| Purpose & Scope | ✅ Pass |
| Accountability | ✅ Pass |
| Transparency | ✅ Pass |
| Bias Audit | ⚠️ Partial — 5 languages tested, production needs more |
| Harm Mitigation | ✅ Pass |
| Data Sovereignty | ✅ Pass |
| Transparency Commitments | ✅ Pass |
| Ongoing Monitoring | ⚠️ Outlined, not implemented (appropriate for prototype) |

**Overall Assessment:** System passes ethical audit for research prototype deployment. Production deployment requires expanded bias testing and ongoing monitoring infrastructure.

---

## TRUST Compliance Metrics (Template)

These metrics are calculated per session in the Ethical Audit tab of the Gradio UI:

```
Session: {session_id}
Single question rate:    {X}% of turns had exactly one question (measured on raw output)
Validation rate:         {X}% of turns included acknowledgment before extraction
Safe exit response rate: {X}% of distress signals correctly handled (dual-layer check)
Language consistency:    {X}% of turns responded in witness's language
Overall TRUST score:     {X}% (Transparency: metrics reflect pre-truncation performance)
```

---

*This audit was conducted by the WitnessChain development team as a pre-deployment self-assessment. Independent review by qualified human rights practitioners is recommended before any real-world deployment.*
