# Prosodic Focus Labels for LLM Conversational Reasoning

This repository contains code, data, and evaluation results for research on the value of prosodic focus information for large language model conversational reasoning.

## Overview

Prosodic focus encodes communicative intent through emphasis, enabling speakers to signal what is pragmatically important in conversation. While prosodic cues are crucial for understanding speaker intent, most automatic speech recognition systems discard this information when producing text transcripts. This work quantifies how much explicit prosodic focus labels improve LLM performance on conversational reasoning tasks.

We evaluate whether focus information helps LLMs generate better follow-up questions that explore the focused element in realistic conversational scenarios (therapy, conflict resolution, interviews, and story continuation). Using a dataset of 150 minimal pairs with oracle focus annotations, we test two models (GPT-4o-mini and DeepSeek Chat) across four scenario types.

## Key Findings

Focus labels consistently improve conversational reasoning across all tested scenarios and models:
- **Absolute improvements:** 27-74 percentage points
- **Relative improvements:** 83-507%
- **Performance with focus:** 60-99% relevance (compared to 14-50% without)

See `data/evaluation_results/COMPREHENSIVE_EVALUATION.md` for detailed results.

## Paper

A detailed paper describing this work is available in (TODO: will add the paper here once i publish it somewhere). The paper evaluates how much explicit prosodic focus labels improve LLM conversational reasoning and establishes the value proposition for automatic focus detection from speech.

## Data Availability

Audio data, minimal pairs, and evaluation results are available via Google Drive:

**Data Access:** [https://drive.google.com/file/d/1fejs5L5u6R-yO3MI4hOLJK1viwczpiTM/view?usp=sharing](https://drive.google.com/file/d/1fejs5L5u6R-yO3MI4hOLJK1viwczpiTM/view?usp=sharing)

The dataset includes:
- 150 minimal pairs with oracle focus annotations
- Synthesized audio with prosodic emphasis
- Evaluation results across 2,400 total evaluations (150 pairs × 2 focus types × 2 models × 4 scenarios)

## Repository Structure

- `scripts/` - Evaluation scripts for conversational reasoning
- `data/` - Evaluation results and data files
- `data/evaluation_results/COMPREHENSIVE_EVALUATION.md` - Detailed evaluation report

## Citation

If you use this work, please cite:

```bibtex
@article{karimi2025prosodicfocus,
  title={The Value of Prosodic Focus Information for Large Language Model Conversational Reasoning},
  author={Karimi, Zalmy},
  year={2025}
}
```

## License

See `LICENSE` for details.

