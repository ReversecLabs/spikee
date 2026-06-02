---
name: goat-questionnaire
description: Guide for creating custom GOAT attack configurations
---

# GOAT Attack Configuration Skill

This skill guides users through creating a custom GOAT attack configuration for their target application.

## Domain
LLM red teaming, prompt injection testing with GOAT (Generative Offensive Adversarial Toolkit) methodology.

## Capabilities
This skill guides users through creating a custom GOAT attack configuration for their target application by:
1. Collecting application information (purpose, users, deployment context)
2. Identifying highest-risk attack objectives
3. Documenting input and output guardrails
4. Mapping adversarial techniques relevant to the use case
5. Generating APPLICATION_CONFIG and APPLICATION_GUARDRAILS Python structures for goat.py

## User Interaction
Users answer a structured questionnaire. The skill then produces a custom *-goat.py file in the workspace attacks/ directory with:
- APPLICATION_CONFIG: Plain-text description of the target application
- APPLICATION_GUARDRAILS: List of guardrail objects with type, location, description, and bypass recommendations

## Common Use Cases
- Creating custom GOAT attacks for specific LLM applications (e.g., banking chatbot, HR assistant)
- Configuring guardrail-bypass tests for security assessments
- Documenting application security controls for red team engagements

## Dependencies
- Requires access to spikee/workspace/attacks/goat.py as the base template
- Generated file follows Spikee attacks module conventions

## Output Format
Generated file: <identifier>-goat.py in workspace/attacks/

## Related Resources
- .github/skills/goat-questionnaire/goat-questionnaire.md (source questionnaire)
- spikee/workspace/attacks/goat.py (template file)
