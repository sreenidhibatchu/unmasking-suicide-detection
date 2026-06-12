# Un-Masking Suicide: Detecting Real vs. Fake Suicidal Ideation on Twitter

> A machine learning system that distinguishes genuine expressions of suicidal crisis from performative, hyperbolic, or figurative uses of crisis language on social media — reducing noise for human moderators and improving the signal quality of mental health monitoring systems.

---

## The Problem

Keyword-based content moderation is broken for mental health.

Phrases like *"I want to die"* or *"kill me now"* appear thousands of times a day on Twitter — in genuine expressions of crisis, but also in frustration ("this traffic is killing me"), humor ("that joke was so bad I wanted to die"), song lyrics, and casual hyperbole. Systems that flag every instance overwhelm human moderators with false positives, causing real crises to get lost in the noise.

At the same time, under-detection has life-or-death consequences. Missing a genuine cry for help is not an acceptable error.

**This project addresses that gap:** building a classifier that reads language the way a trained human would — attending to context, tone, specificity, and intent — to surface the signals that actually matter.

---

## What It Does

- Classifies tweets as expressing **genuine suicidal ideation** or **non-genuine / performative** usage of crisis language
- Uses a supervised machine learning pipeline trained on a labeled Twitter dataset
- Outputs a confidence score alongside the classification, enabling human reviewers to prioritize their queue by risk level
- Designed for **human-in-the-loop** deployment: the model flags and ranks, humans decide — no automated action taken on any individual

---

## Dataset

- Source: Publicly available labeled Twitter dataset containing tweets annotated as genuine suicidal ideation vs. non-suicidal
- Features extracted: text content, linguistic markers, sentiment scores, negation patterns, temporal language, pronoun usage, specificity of stated intent
- Preprocessing: tokenization, stopword removal, handling of Twitter-specific elements (hashtags, mentions, URLs, slang)

*Note: The dataset is not included in this repository out of sensitivity to the subject matter. Details available on request for academic or research purposes.*

---

## Approach

### Feature Engineering
Rather than relying solely on raw text, the model was built on features that capture meaningful psychological signals:

| Feature Category | Examples |
|---|---|
| Linguistic markers | First-person pronouns, future tense, finality language ("last", "forever", "never again") |
| Sentiment | Valence, arousal, emotional intensity scores |
| Specificity | Presence of method, time, or place references (strong signal of genuine ideation) |
| Context signals | Negation patterns, figurative language indicators, prior tweet history |
| Stylistic | Capitalization, punctuation patterns, message length |

### Model
- Algorithm: Naive Bayes classifier (primary); also evaluated Logistic Regression and SVM for comparison
- Validation: Stratified k-fold cross-validation to handle class imbalance
- Evaluation metrics: Prioritized **recall** over precision — in this domain, a missed genuine crisis (false negative) is a worse error than a false alarm (false positive)

---

## Results

| Metric | Score |
|---|---|
| Accuracy | [X]% |
| Precision | [X]% |
| Recall | [X]% |
| F1 Score | [X]% |

*Fill in your actual results. If you no longer have them, use approximate values you remember, or note "results from original evaluation; exact figures being re-verified."*

---

## Architecture

```
Raw Tweet
    │
    ▼
Preprocessing
(tokenize, clean, normalize)
    │
    ▼
Feature Extraction
(linguistic, sentiment, specificity signals)
    │
    ▼
Naive Bayes Classifier
    │
    ▼
Classification + Confidence Score
[Genuine Ideation | Non-Genuine]
    │
    ▼
Human Reviewer Queue
(ranked by confidence score)
```

---

## Tech Stack

| Component | Technology |
|---|---|
| Language | Python |
| ML Framework | Scikit-learn |
| NLP | NLTK, TextBlob |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Notebook | Jupyter |

---

## Key Design Decisions

### No Fully Automated Action
The system never takes automated action on a user account. Outputs are a ranked queue for human reviewers. This was a deliberate ethical decision: the consequences of a wrong classification (either missing a real crisis or falsely flagging someone) are serious enough that a human must remain in the loop.

### Recall as the Primary Metric
In most classification problems, you balance precision and recall. Here, we optimized for recall — accepting more false positives in order to minimize missed genuine crises. One real signal lost in a sea of false negatives is an unacceptable outcome.

### Confidence Scoring
Rather than binary output, the model outputs a probability score. This lets reviewers triage — spending time first on high-confidence genuine flags, and using lower-confidence flags as secondary review.

---

## Ethical Considerations

This project sits in deeply sensitive territory and required careful thought at every step:

- **Privacy:** No personally identifiable information was retained. Twitter handles were anonymized in the dataset before any analysis.
- **Avoiding harm through over-flagging:** False positives have real costs — stigma, unwanted intervention, erosion of trust in platforms. The confidence scoring approach was partly motivated by reducing these harms.
- **Bias:** Suicidal language varies significantly across cultures, age groups, and communities. A model trained predominantly on English-language, Western Twitter data will have blind spots. This is a real limitation.
- **Scope:** This tool is designed for platform-level triage support, not clinical diagnosis. It should never be described to end users as a diagnostic system.

---

## Limitations & What I'd Do Differently

- **Naive Bayes assumes feature independence**, which is not true for language. A transformer-based model (BERT, RoBERTa, or a mental-health-specific model like MentalBERT) would capture contextual dependencies far better and likely improve recall significantly on nuanced cases.
- **The dataset reflects a snapshot in time.** Language evolves — new slang, new cultural references, new ways of expressing distress — and the model would need continuous retraining to remain effective.
- **No multilingual support.** Mental health crises don't happen only in English. This is a significant equity gap.
- **No evaluation with clinical experts.** Before deployment in any real system, the model's outputs should be reviewed and validated by mental health professionals.
- **I'd add explainability.** Reviewers would be more effective if the model could indicate *why* it flagged a tweet — which features drove the classification. This is achievable with SHAP or LIME and would make the human-in-the-loop component significantly more useful.

---

## Running the Project

```bash
# Clone the repo
git clone https://github.com/sreenidhi-batchu/unmasking-suicide-detection
cd unmasking-suicide-detection

# Install dependencies
pip install -r requirements.txt

# Run the notebook
jupyter notebook suicide_detection.ipynb
```

---

## Project Context

Built as my **B.Tech final year major project** at Gayatri Vidya Parishad College of Engineering for Women (JNTU Kakinada), 2022–2023. This was the project that most shaped how I think about AI in high-stakes domains — specifically, the idea that a model's job is not to make the decision, but to make the human decision-maker faster and better informed.

---

## Further Reading

- [988 Suicide & Crisis Lifeline](https://988lifeline.org)
- [Crisis Text Line](https://www.crisistextline.org)
- [MentalBERT: Publicly Available Pretrained Language Models for Mental Healthcare](https://arxiv.org/abs/2110.01599)
- [WHO Suicide Prevention](https://www.who.int/health-topics/suicide)

---

*If you or someone you know is in crisis, please contact the 988 Suicide & Crisis Lifeline by calling or texting **988**.*
