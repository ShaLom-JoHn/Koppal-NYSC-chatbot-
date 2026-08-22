[README.md](https://github.com/user-attachments/files/31329519/README.md)

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
| `_build_notebook.py` | Generates the notebook. Edit this, not the notebook JSON. |
| `_run_notebook_check.py` | Runs every code cell headlessly, to confirm the notebook still works. |

## How the classifier works

Five steps, in the order the notebook does them.

**1. Load.** 1,463 questions, each already labelled with one of 121 topics. Greetings and other
non-questions are dropped. No question appears twice, and no question carries two different labels,
both checked in the notebook.

**2. Split.** 80% to train on, 20% held back to test with. The split is stratified, meaning each
topic keeps the same train/test proportion, so no topic is accidentally missing from training. The
handful of topics with only one question cannot be split, so those go entirely into training.

**3. Top up the thin topics.** 36 topics have fewer than 5 questions, too few to learn from. I
hand-wrote extra phrasings for them, kept in `data/paraphrases.csv`, and added them **to the
training side only**, after the split. Two rules keep the test scores honest: a phrasing is only
added if its topic is already in the training set, and any phrasing that matches a test question
is skipped. 183 were added, 1 was skipped for that reason.

**4. Turn text into numbers.** The model cannot read words, so each question becomes a set of
numbers using TF-IDF, which scores how distinctive each word is. Two versions run side by side:
whole words and word pairs for meaning, and 3 to 5 letter chunks for spelling. The second is what
lets it see "phone" and "phones" as the same thing.

**5. Train and test.** Logistic Regression, with `C=8.0` and `class_weight="balanced"`. Everything
above is wrapped in one pipeline, so the same text processing runs at training time and at
prediction time, with no chance of the two drifting apart.

**Then, at runtime.** The model returns a probability for every topic. If the best one is under
25%, Koppal asks the user to rephrase instead of guessing. That threshold came from testing every
value, not from picking a round number.

## Evaluation results

The model was trained on 1,354 questions and tested on 292 it had never seen. Three tables below:
how well it works, what made it work better, and why it isn't better still.

### 1. How well it works

| What was measured | Score | What that means in plain terms |
| --- | --- | --- |
| Accuracy | 74% | Given 100 new questions, it picks the right topic first time for 74 of them. |
| Macro-F1 | 0.67 | The same idea, but every one of the 121 topics counts equally, so a topic with 3 example questions matters as much as one with 137. It is lower than accuracy because the rare topics are the hard ones. |
| Top-3 accuracy | 90% | The right topic is in its top three guesses 90 times out of 100. So when it is wrong, it is usually only just wrong. |

Accuracy is the number people expect, but on this dataset it flatters the model, because getting
the few big topics right earns most of the score. Macro-F1 is the honest headline, and the reason
it is lower is the third table. Top-3 matters because a chatbot can offer a choice instead of
committing to one answer.

RMSE and MSE do not appear here on purpose. Those measure how far off a predicted *number* is.
This model picks one of 121 categories, so the measures above are the right ones.

### 2. What made it better

| Change made | Macro-F1 |
| --- | --- |
| Starting point | 0.49 |
| Made a mistake on a rare topic cost as much as a mistake on a common one | 0.60 |
| Also let it match on parts of words, not just whole words | 0.67 |

Both changes were one line each. Neither was a bigger or more complex model.

The first was needed because the two largest topics, relocation and posting, were absorbing
questions that belonged to smaller topics. They had the most examples, so guessing them was a safe
bet. Removing that bet is the setting `class_weight="balanced"`.

The second was needed because matching whole words only means "phone" and "phones" look like
different things, and these are real user questions, full of exactly that: "corper" against "corps
member", or "Na must to get two white shoes or one can work?". Matching 3 to 5 letter chunks as
well catches those. That is the character-level TF-IDF vectorizer.

For reference: Complement Naive Bayes actually beat Logistic Regression before any tuning, 0.54
against 0.41. Logistic Regression was still chosen, because the two settings above exist for it
and do not exist for Naive Bayes. Tuned, it finishes well ahead.

### 3. Why it isn't better than that

| Example questions available for a topic | How well the model does on it |
| --- | --- |
| Under 5 | 0.50 |
| 5 to 14 | 0.68 |
| 15 or more | 0.76 |

This is the whole explanation for macro-F1 sitting below accuracy. The model is not weak, it is
under-fed. 74 of the 121 topics sit in that middle row, and the fix is more real questions for
them, not a different model. That is why it is the first item under future work.

### Two other things worth knowing

**It refuses rather than guesses.** The model reports how confident it is, and below 25% Koppal
asks the user to rephrase instead of answering. At that setting it answers about 6 questions in 10
and is right about 4 times in 5 when it does. Raising the bar to 40% would make it right closer to
9 times in 10, but it would then refuse 60% of questions, which is worse behaviour for an
assistant. The full trade-off sweep is in the notebook.

**Its mistakes are near misses, not wild ones.** They cluster inside families of similar topics:
relocation status being read as the general relocation process, PPA questions landing on posting,
batch questions landing on stream assignment. Nothing lands somewhere unrelated. The confusion
matrix in the notebook shows this.

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
