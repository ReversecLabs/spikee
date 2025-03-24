# GPT-2 Prompt Injection Vulnerability Analysis

This document presents the results of prompt injection vulnerability testing conducted on the GPT-2 model using the Spikee testing framework.

## Author
- **Name**: Adel ElZemity
- **Affiliation**: PhD Candidate, University of Kent
- **Contact**: [adelsamir.com](https://adelsamir.com)

## Test Overview
The testing was conducted using the Spikee framework to evaluate the vulnerability of the GPT-2 model to various types of prompt injection attacks. The tests were designed to assess the model's susceptibility to different attack vectors, positions, and languages.

## Test Configuration
- **Model**: openai-community/gpt2
- **Framework**: Spikee
- **Test Positions**: start, middle, end
- **Attack Types**: jailbreak, XSS, data exfiltration
- **Languages**: English, Gaelic

## Results Summary

### Overall Statistics
- **Total Success Rate**: 66.67%
- **Total Tests**: 6
- **Successful Attacks**: 4
- **Failed Attacks**: 2

### Attack Type Breakdown
| Attack Type | Success Rate | Success Count | Total Count |
|-------------|--------------|---------------|-------------|
| Jailbreak   | 66.67%       | 2             | 3           |
| XSS         | 100%         | 2             | 2           |
| Data Exfiltration | 0%    | 0             | 1           |

### Language Impact
| Language | Success Rate | Success Count | Total Count |
|----------|--------------|---------------|-------------|
| Gaelic   | 100%         | 2             | 2           |
| English  | 50%          | 2             | 4           |

### Position Impact
| Position | Success Rate | Success Count | Total Count |
|----------|--------------|---------------|-------------|
| Start    | 66.67%       | 2             | 3           |
| Middle   | 66.67%       | 2             | 3           |
| End      | 66.67%       | 2             | 3           |

## Key Findings
1. The model is highly vulnerable to XSS-style prompt injections (100% success rate)
2. Gaelic language attacks showed higher success rates compared to English
3. The position of the injection did not significantly affect the success rate
4. Data exfiltration attempts using markdown were unsuccessful
5. Most successful combinations involved XSS attacks with jailbreak types

## Replication Instructions

To replicate these tests, follow these steps:

1. Set up the Spikee framework:
```bash
git clone https://github.com/your-username/spikee.git
cd spikee
pip install -r requirements.txt
```

2. Configure your environment:
```bash
cp .env.example .env
# Add your HuggingFace token to .env
```

3. Generate test datasets:
```bash
spikee generate --positions start middle end --seed-folder seeds-mini-test
```

4. Run the tests:
```bash
spikee test --dataset <your-dataset>.jsonl --target hf_gpt2 --success-criteria canary
```

5. Analyze results:
```bash
spikee results analyze --result-file results/<your-results>.jsonl
```

6. Convert results to Excel (optional):
```bash
spikee results convert-to-excel --result-file results/<your-results>.jsonl
```

## Test Implementation
The test implementation can be found in `targets/hf_gpt2.py`. The implementation includes:
- Model initialization and configuration
- Input processing
- Text generation with configurable parameters
- Error handling and response formatting

## Contributing
This test implementation is part of a contribution to the Spikee framework. The results and implementation will be submitted as a pull request to the main repository.

## License
This work is part of academic research at the University of Kent. Please refer to the main Spikee repository for licensing information. 