# NYSC FAQ chatbot

This project is about building a chatbot that answers questions about the Nigerian NYSC (National Youth Service
Corps) scheme: registration, call-up letters, orientation camp, posting, relocation, allowances,
clearance and certificates.

It works in two steps. First a machine learning model reads the question and predicts what the
user is asking about, which is called the **intent**. Then Koppal looks that intent up in a
knowledge base and returns the matching answer. This README covers the classifier, the
evaluation results, and what I would do next.

## Contents

- [What's in the repo](#whats-in-the-repo)
- [How the classifier works](#how-the-classifier-works)
- [Evaluation results](#evaluation-results)
- [What the numbers mean](#what-the-numbers-mean)
- [Known limitations](#known-limitations)
- [How the project got here](#how-the-project-got-here)
- [Future work: Koppal 1.2 and 2.0](#future-work-koppal-12-and-20)
- [How to run it](#how-to-run-it)
- [Licence](#licence)

## What's in the repo

| File | What it is |
| --- | --- |
| `koppal_intent_classifier.ipynb` | The main deliverable. Loads the data, explores it, trains and compares models, tunes the best one, reports the metrics, saves the model, and demos it end to end. |
| `nysc_question_source-1.csv` | The dataset. 1,463 usable questions, each labelled with one of 121 intents. |
| `data/koppal_knowledge_base.csv` | The answers. One row per intent, with a short chat answer and a longer prose answer. |
| `data/paraphrases.csv` | Hand-written paraphrases for the intents that have too few questions. Training set only. |
| `model/koppal_classifier.pkl` | The saved model, text processing and classifier together in one pipeline. |
| `koppal_dialogue_manager.py` | Runtime logic: intent lookup, follow-up branching, slot filling. |
| `koppal_nlu.py` | Loads the saved model and applies the confidence thresholds. |
| `app.py` | Streamlit front end. |
| `koppal_main.py` | Command line loop, used for smoke testing. |

Files starting with `_` are one-off scripts used to build or check things, kept for provenance.
`_build_notebook.py` generates the notebook, so edit that rather than the notebook JSON.

## How the classifier works

**The data.** 1,463 questions across 121 intents. Greetings and other non-questions are dropped
before training. There are no duplicate questions and no question carrying two different labels,
which I checked in the notebook.

**The split.** Stratified 80/20, so each intent keeps roughly the same train/test proportion.
Intents with only one question can't be split, so those go entirely into training. That gives
1,171 training and 292 test questions.

**Augmentation.** 36 intents have fewer than 5 questions, which is too few to learn from. I
hand-wrote paraphrases for them, stored in `data/paraphrases.csv`, and add them **to the
training set only**, after the split. Two guards keep the test scores honest: a paraphrase is
only added if its intent is already in the training set, and any paraphrase that normalises to
the same text as a test question is skipped. 183 were added and 1 was skipped as leakage,
bringing training to 1,354.

**Features.** Two TF-IDF vectorizers joined with `FeatureUnion`:

- word level, unigrams and bigrams, so phrases like "call up letter" are captured
- character level, `char_wb` with 3 to 5 character chunks

The character one matters more than it looks. A word vectorizer treats "phone" and "phones" as
unrelated, and these are real user questions, so they are full of that: "corper" against "corps
member", "tennis" against "sneakers", and plenty of pidgin like "Na must to get two white shoes
or one can work?". Character chunks overlap across all of those.

**The model.** Logistic Regression, `C=8.0`, `class_weight="balanced"`, wrapped in a pipeline
with the vectorizers so the same text processing runs at training time and at prediction time.

**Confidence.** The model returns a probability for every intent. If the top probability is
below 0.25, Koppal asks the user to rephrase instead of guessing. That threshold was chosen from
a sweep, not picked by hand, and the sweep is in the notebook.

## Evaluation results

All numbers are on the held-out test set of 292 questions, which the model never saw during
training. `random_state=42` throughout, so they reproduce.

### Dataset shape

| | |
| --- | --- |
| Questions | 1,463 |
| Intents | 121 |
| Median questions per intent | 8 |
| Largest intent | 137 (`relocation_general_process`) |
| Intents with fewer than 5 questions | 36 |
| Train / test | 1,354 (after augmentation) / 292 |

### Model comparison

Same features, default settings, no tuning.

| Model | Accuracy | Macro-F1 | Weighted-F1 |
| --- | --- | --- | --- |
| Complement NB | 0.685 | 0.539 | 0.629 |
| Logistic Regression | 0.616 | 0.405 | 0.543 |
| Multinomial NB | 0.349 | 0.116 | 0.265 |

Complement NB wins untuned, which makes sense because it is designed for imbalanced text. I
still took Logistic Regression forward, because it has the two settings that fix this dataset's
actual problem and Naive Bayes does not. Tuned, it ends up well ahead.

### Tuning

| Features | `class_weight` | C | Accuracy | Macro-F1 | Top-3 |
| --- | --- | --- | --- | --- | --- |
| word only | none | 8 | 0.644 | 0.492 | 0.860 |
| word only | balanced | 8 | 0.695 | 0.596 | 0.887 |
| word + char | none | 16 | 0.709 | 0.602 | 0.897 |
| **word + char** | **balanced** | **8** | **0.736** | **0.668** | **0.901** |

Two changes did the work, and neither was a bigger model. `class_weight="balanced"` was worth
about +0.10 macro-F1 on its own, and the character n-grams about +0.05, and they stack.

### Final model

| Metric | Score |
| --- | --- |
| Accuracy | 0.736 |
| Macro-F1 | 0.668 |
| Weighted-F1 | 0.720 |
| Log-loss | 1.592 |
| Top-3 accuracy | 0.901 |

### F1 by how much training data the intent had

| Band | Average F1 | Intents |
| --- | --- | --- |
| Rare, under 5 training questions | 0.500 | 6 |
| Medium, 5 to 14 | 0.676 | 74 |
| Well populated, 15 or more | 0.758 | 20 |

This table is the honest explanation for macro-F1 sitting below accuracy, and it points straight
at the next piece of work.

### Choosing the confidence floor

For each possible floor, what share of questions the bot still answers, and how often it is
right when it does answer.

| Floor | Answered | Correct when answered |
| --- | --- | --- |
| 0.00 | 100% | 74% |
| 0.15 | 80% | 78% |
| 0.20 | 72% | 77% |
| **0.25** | **62%** | **81%** |
| 0.30 | 54% | 84% |
| 0.40 | 40% | 88% |

0.25 is the knee. Below it precision flattens out around 77 to 78% without gaining much
coverage. Above it, 0.40 buys a few points of precision but silences a fifth more of the
questions, which is worse behaviour for a chatbot.

### Where it still gets confused

The most common mistakes are all inside families of similar intents: relocation status tracking
and post-relocation process being predicted as the general relocation process, PPA questions
landing on `posting_influence`, and batch questions landing on `stream_assignment`. The full
confusion matrix for the 15 largest intents is in the notebook.

## What the numbers mean

**Accuracy** is how often the top prediction is right. It is the number most people expect, but
it flatters a model on an imbalanced dataset, because getting the big intents right carries most
of the score.

**Macro-F1** averages the F1 score across all 121 intents, counting each one equally. An intent
with 3 questions weighs the same as one with 137. This is the honest headline number here, and
it is lower than accuracy for exactly that reason.

**Weighted-F1** sits between the two, averaging per-intent F1 but weighting by how common each
intent is.

**Log-loss** scores the probabilities rather than the labels, and rewards the model for being
confident when right and uncertain when wrong. It matters here because Koppal uses the
probability to decide whether to answer at all.

**Top-3 accuracy** is how often the correct intent is in the model's top three guesses. At 0.901
against a top-1 of 0.736, this says something useful: when the model is wrong, the right answer
is usually still its second or third choice.

RMSE and MSE do not appear here on purpose. Those are regression metrics, for models predicting a
number. This model predicts one of 121 categories, so the error measures above are the
appropriate ones.

## Known limitations

**Class imbalance is the main one.** 74 of the 121 intents have between 5 and 14 training
questions, and 36 have fewer than 5. The F1-by-band table above shows what that costs.

**The test set is small, and that follows from the dataset size rather than from a choice.** The
split is a standard stratified 80/20, which holds out 292 of the 1,463 questions. The difficulty is
121 intents: even a 20% test set averages only about 2.4 test questions per intent. For a
trustworthy per-intent score you would want at least 10 each, which is 1,210 test questions and
would leave 253 to train on. Going the other way, a 90/10 split gives only 146 test questions and
is noisier still. 80/20 is the least bad point on that curve, so the real fix is more questions,
not a different ratio. Intents with only one question go entirely into training, so those are never
tested at all.

**A single train/test split, chosen for simplicity.** One split is easier to follow in a notebook
meant to be read, and it was enough to compare the three models against each other. Its weakness is
that every number depends on which 292 questions happened to land on the test side. Cross-validation
would evaluate all 1,463 questions, each exactly once across 5 folds, and report a standard
deviation next to the mean, which is what makes small differences interpretable. That is why it is
item 1 of Koppal 1.2 and not a nice-to-have.

**Some intent boundaries are wrong, not just thin.** A few intents score F1 0.00 because a larger
sibling absorbs all their questions. That is a labelling problem rather than a data volume
problem, and it drags macro-F1 down.

**Paraphrases are not real user questions.** They help the model recognise a rare intent's shape,
but they add no genuinely new information, so they are a stopgap for real sourced questions.

## How the project got here

Condensed from `PROJECT_JOURNAL.md`, which has the full reasoning and the source citations.

The dataset had to be built from nothing. No NYSC question dataset existed, I collected
raw questions myself with no structure at first. While collecting I kept running into the same
question worded differently, and that is where the idea of intents came from. I grouped the
questions by intent to make sourcing answers easier. 

The first grouping pass was partly wrong, and matching on words was the reason. "What is a
proper upkeep allowance for a young man?" got filed under `allowance_payment_issue` because it
contained the word allowance, when it is really a general finance question. Grouping by shared
words rather than actual meaning is a mistake I had to go back and undo.

Sourcing answers changed the design. Some questions cannot be answered straight, they need a
follow-up question first. That produced the answer type system, Statement, Procedural or
Conditional, and the branching structure the dialogue manager uses.

Architecture came from reading existing tools Once the structure got too big to hold in my head
I paused the project and researched which brought me to Rasa and Google Dialogflow,
worked out why each of their features existed, and simplified the ideas down to what this project actually 
needed. That became v2. v2 also replaced my assumptions with measurements: I cut some things off and folded some
into each other where they genuinely fit.

Content work found more errors than it found gaps. All 37 camp addresses went in, and Kebbi's
was corrected against a stale third-party source. `term_lookup` turned out to have a real bug where
any correctly classified question would have returned the entire glossary rather than the term
asked about. A definition I had guessed at, PV as "Personal Vetting form", was corrected to Payment
Voucher against two sources. Where no clean answer exists I wrote the caveat instead of a tidy
answer, for example that over 70% of corps members are posted to schools regardless of course.


**This stage was the classifier.** The dataset had fragmented into near-duplicate intents, with
packing alone split across 12 of them, so I merged those down, added paraphrases for the intents
still too thin to learn, and trained the model this README documents.


In summary, the core and toughest part of this project is gathering data (sourcing for question and answers)
and organizing them. Both have to be organic. And lots of decisions that wil likely be changed in future 
still had to be made. The gathering, cleaning, organising, training, auditing loop led to this version.


## Future work: Koppal 1.2 and 2.0

### What this version uses

| Part | Choice |
| --- | --- |
| Features | TF-IDF, word unigrams and bigrams
| Classifier | Logistic Regression, `C=8.0`, `class_weight="balanced"` |
| Validation | Single stratified 80/20 split |
| Rare intents | Hand-written paraphrases, training set only |
| Answer selection | Predicted intent looked up in a CSV knowledge base |
| Uncertainty | Single confidence floor at 0.25, refuse below it |

### Koppal 1.2, incremental improvements to this design

**1. Cross-validation instead of one split.** Do this before anything else. With 5-fold
`StratifiedKFold` and a reported standard deviation, it becomes possible to tell a real
improvement from noise. Right now it is not, which makes every other change on this list hard to
evaluate. This does not raise the score, it makes further work measurable.

**2. Fix the intents that score zero.** `relocation_status_tracking`, `multiple_redeployment`,
`portal_account_access` and `ppa_checking_process` all lose their questions to a bigger sibling.
For each, either write questions that make the distinction visible, or fold it into the parent
the way the 12 packing intents were folded into `camp_kit` and `camp_items_allowed`. Costs no new
data and stops macro-F1 being dragged by intents that cannot win.

**3. Offer the top 3 instead of refusing.** Top-3 accuracy is 0.901 against a top-1 of 0.736, so
roughly 17% of questions have the right answer sitting in second or third place and currently get
thrown away. For confidences between about 0.15 and 0.35, show "did you mean one of these?" with
the three most likely intents. This does not change macro-F1 at all, and it is the single biggest
improvement available to how the bot actually behaves.

**4. Source real questions for the thin intents.** 74 intents sit in the 5 to 14 band, and
bringing them all to 15 is roughly 500 questions. Expected gain is that band moving from 0.676
towards the 0.758 the well-populated intents already reach. Sources that worked before: Nairaland
NYSC threads, NYSC Facebook groups, and the official FAQ. The dataset already has `origin` and
`seed_id` columns for tracking provenance. This is the largest single gain available and also the
slowest.

### Koppal 2.0, changes to the architecture

**5. Sentence embeddings instead of TF-IDF.** Use `sentence-transformers` with a model such as
`all-MiniLM-L6-v2` to turn each question into a vector, then classify with the same Logistic
Regression on top. TF-IDF can only match questions that share literal characters or words, which
is the whole reason paraphrases were needed. An embedding model already knows that "what should I
carry to camp" and "what do I pack for camp" mean the same thing, without being shown an example
of each, so it helps precisely the intents that have too few questions. Costs a model download of
about 90MB and roughly 15 lines of change.

**6. Two-stage classification.** Predict one of the 13 knowledge base categories first, then the
intent within that category. That turns one 121-way decision into a 13-way decision followed by a
roughly 10-way one, and it targets the confusion that actually happens, which is inside families
like relocation and PPA rather than across them. Most work for the least certain gain, so last.

**7. A retrieval layer for questions with no intent.** Every question currently has to land on one
of 121 intents or be refused. Retrieving the closest passages from the knowledge base and
answering from those would cover questions no intent was written for. This is a change in what the
bot is, not a tuning step, which is why it belongs in 2.0.

### Not worth doing

**More paraphrases for intents that already have enough.** I measured this. Adding phrasing
variants to `camp_kit` and `camp_items_allowed`, which already had 40 and 20 questions, cost about
0.008 macro-F1 each while fixing real routing failures. Worth it for those two specific cases,
not worth it as a general strategy. Paraphrases add wording, not information.

**Chasing macro-F1 past about 0.70 on this dataset.** With 292 test questions across 121 intents,
improvements smaller than about 0.03 cannot be distinguished from noise. Fix the validation
(item 1) before trusting any number in that range.

## How to run it

The notebook is self-contained. Open `koppal_intent_classifier.ipynb` and Run All. It trains from
the CSVs, prints every metric above, saves the model to `model/koppal_classifier.pkl`, and
finishes with a live demo.

Requirements are in `requirements.txt`: `pandas`, `scikit-learn`, `matplotlib`, `joblib`.

To rebuild the notebook after changing it, edit `_build_notebook.py` and run it, then verify with
`_run_notebook_check.py`, which executes every code cell headlessly.

## Licence

This repository is covered by two separate licences, because the code and the data are not the
same kind of work.

**Code, under Apache License 2.0.** That is the notebook, `_build_notebook.py`,
`_run_notebook_check.py`, and any other source file here. See `LICENSE` in the repository root.
Apache-2.0 was chosen over MIT for two reasons: it includes an express patent grant, and it
requires anyone who modifies the code to say that they did.

**Data, all rights reserved, permission required.** That is `nysc_question_source-1.csv`,
`data/koppal_knowledge_base.csv` and `data/paraphrases.csv`. See `data/LICENSE`. No copying,
redistribution, or use as training data for any released model without prior written permission.
Reading it and running the notebook locally is allowed, and so is academic assessment of this
submission, and nothing beyond that is.

The dataset is licensed this way because it is the part of this project that took the most work
and did not previously exist. No NYSC question dataset was available, so the questions were
collected, cleaned, deduplicated and labelled into 121 intents by hand, and every answer was
sourced and written. The compilation, the taxonomy, the labels and the answer text are original
work and the licence protects them as such.

Note that a patent grant only makes sense for code, since a dataset is not patentable subject
matter. The dataset is protected instead by copyright and compilation rights, which is what
`data/LICENSE` asserts.
