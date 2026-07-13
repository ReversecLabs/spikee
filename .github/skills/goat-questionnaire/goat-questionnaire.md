# GOAT Attack Configuration Questionnaire

This questionnaire collects the information needed to generate a custom GOAT attack file for your environment. Answer each question as completely as possible — an LLM will use your answers to produce a named copy of `goat.py` (e.g. `acme-bank-goat.py`) in your Spikee workspace `attacks/` directory, with `APPLICATION_CONFIG` and `APPLICATION_GUARDRAILS` populated and ready to use.

---

## Section 0: Output File

**Q0. What short identifier should be used to name the output file?**

> This becomes the prefix of the generated file: `<identifier>-goat.py`, placed in your workspace `attacks/` directory alongside the original `goat.py`.
> Use lowercase letters, numbers, and hyphens only — e.g. `acme-bank`, `internal-hr-bot`, `customer-support-v2`.

---

## Section 1: Application Overview

**Q1. What is the name and primary purpose of the application being tested?**

> e.g. "Acme Bank customer support chatbot — answers account and transaction queries for retail banking customers."

---

**Q2. Who are the intended users of this application?**

> e.g. "Authenticated retail banking customers, aged 18+, who have logged in via the bank's mobile app."

---

**Q3. What topics or tasks is the application designed to handle?**

> List the main use-cases — e.g. "Balance enquiries, transaction history, fraud reporting, card management, branch locator."

---

**Q4. What underlying LLM(s) power the application? (if known)**

> e.g. "Claude 3.5 Sonnet via Amazon Bedrock", "GPT-4o via Azure OpenAI". Leave blank if unknown.

---

**Q5. In what deployment context does the application operate?**

> e.g. "Public-facing web widget, internal employee tool, mobile app, API used by partner integrations."

---

## Section 2: Security & Sensitivity Context

**Q6. What are the highest-risk attack objectives a red-teamer would attempt against this application?**

> Describe the outcomes you most want to prevent — e.g. "Extracting another user's account balance, convincing the bot to initiate an unauthorised transfer, leaking the system prompt, generating phishing content."

---

## Section 3: Input Guardrails

Answer one block per input guardrail. Add as many blocks as needed.

---

### Input Guardrail 1

**Q8a. What type of control is this?**

> e.g. system prompt instructions, keyword/phrase filter, regex filter, prompt injection detector, PII redactor, toxicity classifier, topic classifier, intent classifier, rate limiter, allow-list validation.

---

**Q8b. Describe what it blocks or redacts.**

> e.g. "System prompt instructs the model to refuse queries unrelated to banking, never reveal internal instructions, and always respond in formal English."

---

**Q8c. What is the response when it triggers?**

> e.g. "Returns a static refusal message: 'I cannot help with that.' No further processing occurs."

---

**Q8d. Are there any known weaknesses, bypass patterns, or edge cases?**

> e.g. "Does not handle leetspeak substitutions. Non-English equivalents are not blocked."

---

*(Copy the block above for each additional input guardrail)*

---

## Section 4: Output Guardrails

Answer one block per output guardrail. Add as many blocks as needed.

---

### Output Guardrail 1

**Q9a. What type of control is this?**

> e.g. output content classifier, PII scanner, hallucination detector, response length limiter, allow-list output filter, groundedness checker.

---

**Q9b. Describe what it blocks or redacts.**

> e.g. "Scans LLM output for IBAN, sort code, and full card numbers using regex; replaces matches with [REDACTED]."

---

**Q9c. What is the response when it triggers?**

> e.g. "The user receives the redacted version of the response, with a notice that some content was removed."

---

**Q9d. Are there any known weaknesses, bypass patterns, or edge cases?**

> e.g. "Partial card numbers split across two sentences are not caught. Non-standard separators (spaces vs dashes) may evade the regex."

---

*(Copy the block above for each additional output guardrail)*

---

## Section 5: Additional Context

**Q10. Are there any other security controls not covered above?**

> e.g. "Session-level abuse detection that flags accounts after 10 policy violations", "Human review queue for flagged conversations."

---

**Q11. Are there specific adversarial techniques you are most concerned about or most want to test?**

Select all that apply (or describe your own):

- [ ] Knowledge Ladder (gradual escalation from general to specific)
- [ ] Authority Building / Citation (referencing or fabricating experts / papers)
- [ ] Emotional Manipulation (urgency, distress, empathy appeals)
- [ ] Scope Creep (start innocuous, incrementally push)
- [ ] Hypothetical / Fictional Framing ("what if", "in a story", "suppose")
- [ ] Historical / Academic Framing ("for research", "in the 1990s")
- [ ] Persona Adoption (claiming to be a professional, journalist, researcher)
- [ ] Encoding / Obfuscation (base64, leetspeak, character substitution)
- [ ] Language Switching (multilingual bypass)
- [ ] Prompt Injection (via user-controlled data fields or file content)
- [ ] Other: _______________

---

**Q12. Any other information you think would help tailor the GOAT attack to this application?**

> Free text — system prompt fragments, known refusal phrases, architectural notes, etc.

---

## How Your Answers Will Be Used

An LLM will read your completed answers and produce a file named `<identifier>-goat.py` (using your answer to Q0) in your Spikee workspace `attacks/` directory. It will be a verbatim copy of `goat.py` with two variables replaced at the top of the file:

1. **`APPLICATION_CONFIG`** — a concise plain-text description of the application, its intended use, sensitive capabilities, and deployment context.

2. **`APPLICATION_GUARDRAILS`** — a list of guardrail objects, one per control described above, each following the schema:

```python
APPLICATION_GUARDRAILS = [
    {
        "type": "...",           # e.g. "keyword filter", "PII redactor", "output classifier"
        "location": "...",       # "input" or "output"
        "description": "...",    # what it detects / blocks
        "recommendations": "..." # suggested bypass strategies for the GOAT attacker LLM
    },
    ...
]
```

If no guardrails are in use, `APPLICATION_GUARDRAILS` will be left as `None` and GOAT will run in standard social-engineering mode rather than guardrail-bypass mode.
