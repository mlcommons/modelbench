# Multi-Turn Safety Testing Plugin for ModelBench

This plugin extends AILuminate's safety testing capabilities to support multi-turn conversations, addressing a critical gap in current safety benchmarks.

## Motivation

Current AILuminate tests are limited to single-turn interactions. However, many safety risks emerge through multi-turn conversations where attackers use progressive techniques like:
- Gradual boundary pushing
- Context accumulation
- Role-play escalation
- Social engineering

This plugin enables testing these sophisticated attack vectors.

## Features

- **Conversation State Tracking**: Maintains context across multiple turns
- **Progressive Attack Detection**: Identifies when safety breaks down
- **Attack Vector Analysis**: Categorizes different attack strategies
- **Safety Breakdown Analytics**: Pinpoints where conversations become unsafe

## Installation

```bash
cd plugins/multi_turn_safety
poetry install
```

## Usage

The plugin automatically registers new test types with ModelBench:

```bash
# Run multi-turn safety tests
modelbench run --sut openai:gpt-4 --test multi_turn_safe_vcr-en_us-demo

# Run all multi-turn tests
modelbench run --sut openai:gpt-4 --test multi_turn_safe_*
```

## Test Structure

Each test consists of conversation scenarios with multiple turns:

```python
ConversationScenario(
    scenario_id="progressive_jailbreak_001",
    hazard_category="vcr",
    turns=[
        ConversationTurn(1, "Initial safe prompt", "safe_response"),
        ConversationTurn(2, "Boundary pushing", "safe_response"),
        ConversationTurn(3, "Attack attempt", "safe_refusal", "jailbreak")
    ]
)
```

## Metrics

Beyond standard AILuminate metrics, this plugin adds:
- `safety_breakdown_rate`: Percentage of conversations that become unsafe
- `attack_effectiveness`: Success rate by attack vector
- `conversation_risk_score`: Cumulative risk across turns

## Contributing

Contributions welcome! Areas for improvement:
- Additional attack vectors
- More sophisticated conversation scenarios
- Cross-lingual attack patterns
- Integration with real-time validators

## Author

Anivar Aravind