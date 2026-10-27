# Classifier 3: Zero-Shot LLM (Claude)

Classifies each review with a single call to the Anthropic Claude API. There is no training and the model sees no labeled examples. We test whether general language understanding can stand in for supervised learning, and how much the prompt wording matters.

## Method

| Setting | Value |
|---------|-------|
| Model | `claude-haiku-4-5-20251001` |
| Max output tokens | 20 |
| Eval set | Stratified sample of the test set: 500 reviews per condition = **3,000 reviews** (`random_state=42`) |
| Parsing | Substring match against the 6 condition names. Anything else is labeled `unknown` and counted as wrong. |
| Cost | Under about $1 in API costs for the full run |

### Prompt variants

**Prompt A (minimal):** lists the conditions and asks for the condition name only.

**Prompt B (descriptive):** adds three things:
- **Expert framing:** tells the model it is an expert medical text classifier.
- **Guidelines:**
  - Focus on symptoms, drugs and outcomes.
  - Don't rely on the drug name alone.
  - If several conditions are mentioned, pick the one the drug is mainly prescribed for.
- **Strict format instruction:** asks for the condition name only, with no explanation or extra text.

Both prompts are defined in [`zero_shot_llm.py`](zero_shot_llm.py).

## Run

From the repo root:

```bash
export ANTHROPIC_API_KEY=your_key_here           # never commit your key

python 03_zero_shot_llm/zero_shot_llm.py --sanity-check      # 5 reviews, quick test
python 03_zero_shot_llm/zero_shot_llm.py                     # full run, about 6,000 API calls
python 03_zero_shot_llm/zero_shot_llm.py --n-per-class 50    # cheaper run
python 03_zero_shot_llm/zero_shot_llm.py --prompts b         # just Prompt B
```

Outputs are written to `results/zero_shot_llm/`:
- `zero_shot_results.csv`: one row per review, with the predicted label and the model's raw reply for each prompt
- `confusion_matrix_prompt_*.csv`

## Results (n = 3,000)

| Prompt | Accuracy | Macro F1 | Unknowns |
|--------|---------:|---------:|---------:|
| A (minimal) | 77% | 0.77 | 422 |
| B (descriptive) | 90% | **0.91** | 49 |

| Condition | F1 (A) | F1 (B) |
|-----------|-------:|-------:|
| ADHD | 0.94 | 0.95 |
| Acne | 0.93 | 0.94 |
| Anxiety | 0.84 | 0.84 |
| Birth Control | 0.91 | 0.93 |
| Depression | 0.84 | 0.84 |
| Diabetes, Type 2 | **0.18** | **0.96** |

**Takeaways**
- Rewording the prompt raised macro F1 by **14 points**. That is more than the gap between SVM and BERT.
- Prompt B gets within about 3 F1 points of fine-tuned BERT with no training data and no GPU.
- Almost all of Prompt A's loss comes from **Diabetes, Type 2**, where recall dropped to 0.10.

> **Note on Prompt A's unknowns:** parsing requires the exact string `"Diabetes, Type 2"`. A reply such as `"Diabetes Type 2"` (no comma) is counted as `unknown`. This likely explains many of Prompt A's 422 unknowns and its low Diabetes recall. Prompt B's stricter format instruction ("no punctuation or additional text") cuts down on those near-misses. The script saves each raw reply so you can check this yourself.

