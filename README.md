# Banking Intent Classifier: fine-tuning a small language model with LoRA

Fine-tuned **Qwen2.5-0.5B-Instruct** to classify customer banking queries into 77 intents and reply in strict JSON, e.g. `{"intent": "card_arrival"}`.

**Model:** https://huggingface.co/Kishorekumar32/qwen2.5-0.5b-banking77-intent-lora |
**Dataset:** https://huggingface.co/datasets/Kishorekumar32/banking77-intent-chat

## Results (official Banking77 test set, 3,076 examples)

| Metric | Untuned base model* | Fine-tuned |
|---|---|---|
| Accuracy | 17.5% | **92.7%** |
| Macro-F1 | not measured | 0.893 |
| Valid intent label | 76.5% | 99.9% |

*Base model measured on a fixed random 200-example subset, with all 77 intent names given in the prompt. The fine-tuned model scored 91.5% on the same subset, using a shorter prompt with no label list.

## What I did

1. **Data quality checks:** missing values, duplicates, conflicting labels, train/test leakage, class balance, token lengths (all clean).
2. **Stratified 90/10 train/validation split**, with the official test set held out until the end.
3. **Chat-format conversion** (system / user / assistant) with JSON answers.
4. **Baseline evaluation** of the untuned model before training.
5. **LoRA fine-tuning** (r=16, alpha=32, all linear layers, 1.75% of parameters trainable), loss on the answer only, 2 epochs, about 11.5 minutes on a free Colab T4.
6. **Evaluation and error analysis:** most errors are between near-synonymous intents (e.g. `card_arrival` vs `card_delivery_estimate`).
7. **Published** the adapter, dataset, and model card to Hugging Face, then verified by reloading from the Hub.

## Limitations

Short English queries only. The model always picks one of the 77 known intents, even for out-of-scope input. Not production-validated.

## Tools

Python, PyTorch, Hugging Face Transformers, PEFT, TRL, Datasets, scikit-learn, Google Colab. Built with AI-assisted coding.

Author: Kishore kumar | LinkedIn: https://www.linkedin.com/in/kishore-kumar-telugu/?isSelfProfile=true
