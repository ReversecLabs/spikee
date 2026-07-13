---
name: target-implementation
description: Guide for implementing Spikee target modules
---

# Spikee Target Implementation Skill

This skill guides users through implementing a Spikee target module.

## Domain
Creating Spikee target modules for testing LLMs, LLM applications, and guardrails.

## Capabilities
This skill guides users through implementing a Spikee Target module by:
1. Determining target type (LLM vs Guardrail vs Multi-turn)
2. Selecting appropriate base class (Target, SimpleMultiTarget, or MultiTarget)
3. Implementing required methods (get_description, get_available_option_values, process_input)
4. Configuring authentication patterns (API keys, OAuth2, Managed Identity, GCP ADC, etc.)
5. Handling guardrail triggers and response parsing
6. Configuring proxy and TLS options

## User Interaction
The skill helps users:
1. Gather target system information (HTTP request/response samples, authentication details)
2. Answer a decision tree to select the correct base class
3. Match their authentication/transport pattern to Spikee templates
4. Implement the target with all required error handling

## Common Use Cases
- Creating targets for custom LLM APIs (OpenAI-compatible and native SDKs)
- Implementing multi-turn chatbot targets with session management
- Building guardrail targets that return bool (True=bypassed, False=blocked)
- Adding authentication handlers (API keys, bearer tokens, OAuth2, Azure IMDS, GCP ADC)

## Dependencies
- Requires Spikee installed with target templates
- Follows spikee.templates module conventions

## Available Base Classes
| Scenario | Base Class | Import |
|----------|------------|--------|
| Single-turn LLM or guardrail | Target | from spikee.templates.target import Target |
| Multi-turn, spikee-managed history | SimpleMultiTarget | from spikee.templates.simple_multi_target import SimpleMultiTarget |
| Multi-turn, custom state | MultiTarget | from spikee.templates.multi_target import MultiTarget |

## Authentication Patterns Supported
1. Static API Key / Bearer Token
2. Static Session Cookie
3. Username / Password (Form Login)
4. Azure Managed Identity (IMDS)
5. GCP Application Default Credentials
6. Auth0 / OAuth2 Refresh Token
7. Internal JWT (Self-Fetched)

## Response Handling
- LLM targets: Return string (LLM reply), raise GuardrailTrigger for blocks
- Guardrail targets: Return bool (True=bypassed, False=blocked)

## Output Format
Generated file: {application}_{feature}.py in workspace/targets/

## Related Resources
- .github/skills/target-implementation/target-instructions.md (source documentation)
- spikee/templates/target.py (base class template)
- spikee/templates/simple_multi_target.py (multi-turn template)
- spikee/templates/multi_target.py (complex multi-turn template)
