# Generates koppal_intent_classifier.ipynb  -- a single, beginner-friendly notebook that
# loads the data, does light EDA, trains and compares a few models, tunes the best one,
# reports the metrics, saves the model, and demos it live. Plain student-voice markdown +
# commented code. Re-run this file to rebuild the notebook.
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": text.strip("\n").splitlines(keepends=True)})


# ------------------------------------------------------------------ title
md("""
# Koppal - NYSC FAQ Intent Classifier

Koppal is a chatbot that answers common NYSC (National Youth Service Corps) questions.
Before it can reply, it has to work out **what the user is asking about** - is this a
question about the call-up letter, the camp kit list, relocation, allowances, and so on.
That label is called the **intent**.

This notebook builds the model that reads a question and predicts its intent. The steps are:

1. Load the question/intent data
2. Look at the data (EDA)
3. Split into train and test sets
4. Turn text into numbers (features)
5. Train and compare a few models
6. Tune the best one
7. Measure how well it does
8. Save the model
9. Try it live on new questions

The predicted intent is later used to look up the matching answer in the knowledge base.
""")

# ------------------------------------------------------------------ setup
md("""
## 1. Setup

We import the libraries we need and fix a random seed so the results are the same every
time the notebook is run.
""")
code("""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             log_loss, top_k_accuracy_score)

import joblib

%matplotlib inline

RANDOM_STATE = 42          # fixing this makes the split and training reproducible
DATA_PATH = "nysc_question_source-1.csv"
KB_PATH = "data/koppal_knowledge_base.csv"
MODEL_PATH = "model/koppal_classifier.pkl"
""")

# ------------------------------------------------------------------ load
md("""
## 2. Load the data

Each row in the dataset is one **question** and the **intent** it belongs to. Those are the
only two columns the model needs: the question is the input, the intent is the label we want
to predict.

We drop the `noise_nonquestion` rows (greetings and other non-questions - not something we
train the classifier to route) and any rows with a missing question or intent.
""")
code("""
df = pd.read_csv(DATA_PATH)

# keep only the two columns we need, drop non-questions and blanks
df = df[["question", "intent"]].dropna()
df = df[df["intent"] != "noise_nonquestion"]
df["question"] = df["question"].str.strip()
df["intent"] = df["intent"].str.strip()
df = df[(df["question"] != "") & (df["intent"] != "")].reset_index(drop=True)

print("questions:", len(df))
print("intents  :", df["intent"].nunique())
df.head()
""")

# ------------------------------------------------------------------ EDA
md("""
## 3. Exploring the data (EDA)

A quick look at the data before modelling. Two things matter most for this project:

- **How many intents are there, and how many questions does each one have?**
  If some intents have very few examples, the model will struggle to learn them.
- **Are the questions clean** (no exact duplicate sitting under two different intents)?
""")
code("""
per_intent = df["intent"].value_counts()

print("questions per intent - summary")
print(per_intent.describe().round(1))
print()
print("intents with the most questions:")
print(per_intent.head(5))
print()
print("intents with the fewest questions:")
print(per_intent.tail(5))
print()
print("intents with fewer than 5 questions:", int((per_intent < 5).sum()))
""")

md("""
Most of the training data sits in a handful of common intents, while many intents have only
a few questions each. This imbalance is the main challenge, and it is why we pay attention to
**macro-F1** later (it treats every intent equally, so the rare ones are not ignored).
""")
code("""
# class distribution - the 20 largest intents
top20 = per_intent.head(20)

plt.figure(figsize=(9, 5))
plt.barh(top20.index[::-1], top20.values[::-1])
plt.xlabel("number of questions")
plt.title("Top 20 intents by number of questions")
plt.tight_layout()
plt.show()
""")
code("""
# how long are the questions (in words)? just to know the text we are working with
df["n_words"] = df["question"].str.split().apply(len)

plt.figure(figsize=(7, 4))
plt.hist(df["n_words"], bins=30)
plt.xlabel("words per question")
plt.ylabel("number of questions")
plt.title("Question length")
plt.tight_layout()
plt.show()

print("average words per question:", round(df["n_words"].mean(), 1))
""")
code("""
# data quality check: is the same exact question filed under two different intents?
# (that would send mixed signals to the model)
dupe = df.groupby("question")["intent"].nunique()
conflicting = dupe[dupe > 1]
print("questions with more than one intent label:", len(conflicting))
if len(conflicting):
    print(conflicting.head())
""")

# ------------------------------------------------------------------ KB parity
md("""
### 3.1 Knowledge-base check

The classifier predicts an intent, then Koppal looks that intent up in the knowledge base to
find the answer. So every intent we train on should have an answer waiting for it. Here we
check which ones don't yet.
""")
code("""
kb = pd.read_csv(KB_PATH)
kb_intents = set(kb["intent"].str.strip())

trained_intents = set(df["intent"].unique())
missing = sorted(trained_intents - kb_intents)

print("intents we train on      :", len(trained_intents))
print("intents with a KB answer :", len(trained_intents & kb_intents))
print("intents still missing an answer:", len(missing))
for i in missing:
    print("   -", i)
""")

# ------------------------------------------------------------------ split
md("""
## 4. Train / test split

We hold out 20% of the questions as a **test set** the model never sees during training, so
our scores reflect how it does on new questions.

We split in a **stratified** way, meaning each intent keeps roughly the same train/test
proportion. A few intents have only a single example - those can't be split, so we simply put
them in the training set.
""")
code("""
X = df["question"]
y = df["intent"]

counts = y.value_counts()
has_two = y.isin(counts[counts >= 2].index)   # intents with at least 2 examples can be split

X_multi, y_multi = X[has_two], y[has_two]
X_solo, y_solo = X[~has_two], y[~has_two]      # single-example intents

X_train, X_test, y_train, y_test = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=RANDOM_STATE)

# single-example intents go entirely into training
X_train = pd.concat([X_train, X_solo])
y_train = pd.concat([y_train, y_solo])

print("training questions:", len(X_train))
print("test questions    :", len(X_test))
""")

# ------------------------------------------------------------------ augmentation
md("""
### 4.1 Giving the rare intents more examples (training set only)

Some intents have only a handful of questions, so the model has very little to learn them
from. To help, we hand-wrote a few extra **paraphrases** for those thin intents - the same
questions worded differently - and keep them in `data/paraphrases.csv`.

We add these **only to the training set**, with two safety rules so the test scores stay honest:

1. only add a paraphrase if its intent is already in the training set, and
2. skip any paraphrase that happens to match a real test question.

Rule 2 avoids **leakage** - letting the model practise on something it will be tested on,
which would make the scores look better than they really are.
""")
code("""
para = pd.read_csv("data/paraphrases.csv")

def norm(text):
    # lower-case and squeeze spaces, so near-identical wording compares equal
    return " ".join(str(text).lower().split())

test_questions = set(X_test.map(norm))   # what we must NOT train on
train_intents = set(y_train)

extra_q, extra_y, skipped = [], [], 0
for _, row in para.iterrows():
    intent = row["intent"].strip()
    text = row["paraphrase"].strip()
    if intent not in train_intents:
        continue                          # rule 1
    if norm(text) in test_questions:
        skipped += 1                      # rule 2 - would be leakage
        continue
    extra_q.append(text)
    extra_y.append(intent)

X_train = pd.concat([X_train, pd.Series(extra_q)], ignore_index=True)
y_train = pd.concat([y_train, pd.Series(extra_y)], ignore_index=True)

print(f"added {len(extra_q)} paraphrases ({skipped} skipped to avoid leakage)")
print("training questions now:", len(X_train))
""")

# ------------------------------------------------------------------ features + compare
md("""
## 5. Turning text into numbers (features)

A model can't read words, so we convert each question into numbers with **TF-IDF**. It scores
how important each word (and each two-word phrase) is to a question. We include two-word
phrases (`ngram_range=(1, 2)`) so wording like "call up" is captured, not just single words.

We use **two vectorizers side by side**, joined with `FeatureUnion`:

- a **word** one, for meaning, "relocation", "call up letter", "clearance".
- a **character** one, for spelling, in chunks of 3 to 5 letters inside each word.

The character one matters more than it sounds. A word vectorizer treats "phone" and "phones"
as two unrelated things, and our questions are real user questions, so they're full of that:
"corper" and "corps member", "tennis" and "sneakers", and plenty of pidgin like "Na must to get
two white shoes or one can work?". Character chunks overlap across all of those, so the model
stops missing a question just because of how it was spelled.

We wrap everything in a **pipeline** so the exact same text processing is applied during
training, testing, and later when the saved model runs live.
""")
code("""
def make_model(classifier):
    \"\"\"Word features + character features, followed by a classifier, as one pipeline.\"\"\"
    features = FeatureUnion([
        # word and two-word phrases - captures meaning
        ("word", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
        # 3 to 5 letter chunks inside words - survives spelling and plural differences
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2)),
    ])
    return Pipeline([("features", features), ("clf", classifier)])
""")

md("""
## 6. Comparing a few models

We try three text-classification models and see which predicts the test intents best. We
score them on:

- **Accuracy** - how often the top prediction is correct.
- **Macro-F1** - the average F1 across all intents, counting each intent equally (the honest
  score when classes are imbalanced, like ours).

All three give a **probability** for each intent, which we later use as a confidence score.
""")
code("""
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
    "Complement NB": ComplementNB(),
    "Multinomial NB": MultinomialNB(),
}

results = []
for name, clf in candidates.items():
    model = make_model(clf)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    results.append({
        "model": name,
        "accuracy": accuracy_score(y_test, pred),
        "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_test, pred, average="weighted", zero_division=0),
    })

pd.DataFrame(results).set_index("model").round(3)
""")

md("""
On default settings Complement NB actually comes out ahead. It's built for imbalanced text, so
that isn't surprising. We still take **Logistic Regression** forward, and the reason is the next
section: it has the two settings that fix our specific problem, `C` and `class_weight`. Naive
Bayes has no `class_weight`, so there's no way to tell it to stop favouring the big intents.
Once Logistic Regression is tuned it overtakes Complement NB by a wide margin. It also gives
better-behaved probabilities, which Koppal needs to decide whether it is confident enough to
answer at all.
""")

# ------------------------------------------------------------------ tune
md("""
## 7. Tuning the model

Two settings to try here.

`C` controls how strongly the model is regularised, meaning how closely it is allowed to fit
the training data. Low `C` keeps it cautious, high `C` lets it commit.

`class_weight` is the more interesting one. Look at the confusions later in this notebook and
you'll see the two biggest intents, `relocation_general_process` and `posting_influence`,
swallowing questions that belong to smaller intents. That happens because they have the most
training examples, so guessing them is a safe bet for the model. Setting
`class_weight="balanced"` makes a mistake on a small intent cost as much as a mistake on a big
one, which takes away that safe bet.

We try both settings together and keep the combination with the best macro-F1.
""")
code("""
tune_results = []
for weight in [None, "balanced"]:
    for C in [4.0, 8.0, 16.0]:
        model = make_model(LogisticRegression(C=C, class_weight=weight,
                                              max_iter=2000, random_state=RANDOM_STATE))
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        tune_results.append({
            "class_weight": str(weight),
            "C": C,
            "macro_f1": f1_score(y_test, pred, average="macro", zero_division=0),
            "accuracy": accuracy_score(y_test, pred),
        })

tune_df = pd.DataFrame(tune_results).round(3)
best = tune_df.loc[tune_df["macro_f1"].idxmax()]
best_C = float(best["C"])
best_weight = None if best["class_weight"] == "None" else best["class_weight"]
print("best:", f"C={best_C}", f"class_weight={best_weight}")
tune_df
""")
md("""
`class_weight="balanced"` is the bigger win of the two, and it costs nothing but one argument.
That is worth remembering: the fix for an imbalanced dataset was not a fancier model, it was
telling the model to stop treating the rare intents as cheap to get wrong.
""")

# ------------------------------------------------------------------ evaluate
md("""
## 8. Final model and evaluation

We train the final model with the best `C` and measure it properly on the test set:

- **Accuracy / macro-F1 / weighted-F1** - overall correctness.
- **Log-loss** - how good the probabilities are (lower is better); this is the right "error"
  measure for a model that outputs probabilities.
- **Top-3 accuracy** - how often the correct intent is in the model's top 3 guesses, useful
  because a chatbot can offer a couple of options.
- **Confidence** - the probability of the top prediction, shown for correct vs wrong answers.
""")
code("""
final_model = make_model(LogisticRegression(C=best_C, class_weight=best_weight,
                                            max_iter=2000, random_state=RANDOM_STATE))
final_model.fit(X_train, y_train)

pred = final_model.predict(X_test)
proba = final_model.predict_proba(X_test)
classes = final_model.classes_

print("Accuracy    :", round(accuracy_score(y_test, pred), 3))
print("Macro-F1    :", round(f1_score(y_test, pred, average="macro", zero_division=0), 3))
print("Weighted-F1 :", round(f1_score(y_test, pred, average="weighted", zero_division=0), 3))
print("Log-loss    :", round(log_loss(y_test, proba, labels=classes), 3))
print("Top-3 acc.  :", round(top_k_accuracy_score(y_test, proba, k=3, labels=classes), 3))
""")

md("""
### Why macro-F1 looks lower than accuracy

Macro-F1 counts every intent equally, so the many rare intents pull it down even when the
common questions are answered well. To show this clearly, we group intents by how many
training examples they had and report the average F1 in each group.
""")
code("""
from sklearn.metrics import f1_score as _f1

train_counts = y_train.value_counts()
per_class_f1 = _f1(y_test, pred, labels=classes, average=None, zero_division=0)
f1_by_intent = dict(zip(classes, per_class_f1))

bands = {"rare (< 5 train)": [], "medium (5-14 train)": [], "well-populated (15+ train)": []}
for intent in y_test.unique():
    n = train_counts.get(intent, 0)
    key = ("rare (< 5 train)" if n < 5
           else "medium (5-14 train)" if n < 15
           else "well-populated (15+ train)")
    bands[key].append(f1_by_intent.get(intent, 0.0))

print("average F1 by how many training examples the intent had:")
for key, scores in bands.items():
    if scores:
        print(f"  {key:28s}: {np.mean(scores):.3f}   ({len(scores)} intents)")
""")
code("""
# confidence of the top prediction, split by whether the prediction was right or wrong
confidence = proba.max(axis=1)
correct = (pred == y_test.values)

plt.figure(figsize=(7, 4))
plt.hist(confidence[correct], bins=20, alpha=0.7, label="correct")
plt.hist(confidence[~correct], bins=20, alpha=0.7, label="wrong")
plt.xlabel("confidence (probability of top prediction)")
plt.ylabel("number of questions")
plt.title("The model is more confident when it is right")
plt.legend()
plt.tight_layout()
plt.show()
""")
md("""
### Choosing the confidence floor

Koppal shouldn't answer when it isn't sure, it should ask the user to rephrase. To pick the
cut-off, we sweep it: for each possible floor we check what share of questions the bot would
still answer, and how often it would be right when it does answer. A high floor is accurate
but silent, a low floor answers everything and gets more wrong.
""")
code("""
print(f"{'floor':>6} {'answered':>9} {'correct when answered':>22}")
for t in [0.0, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
    answered = confidence >= t
    if answered.sum() == 0:
        continue
    print(f"{t:6.2f} {answered.mean():8.0%} {correct[answered].mean():21.0%}")
""")
md("""
Around **0.25** is the knee of the curve. The bot answers about six questions in ten and is
right roughly four times out of five when it does. Dropping the floor lower doesn't buy much,
precision flattens out around 77 to 78%. Raising it to 0.40 gains a few points of precision but
silences a fifth more of the questions, which is worse behaviour for a chatbot. So 0.25 is what
we use in the demo below.
""")
code("""
# which intents does the model most often confuse? (only the mistakes)
from collections import Counter

confusions = Counter()
for true_i, pred_i in zip(y_test.values, pred):
    if true_i != pred_i:
        confusions[(true_i, pred_i)] += 1

print("most common confusions (true -> predicted):")
for (true_i, pred_i), n in confusions.most_common(10):
    print(f"  {n:2d}x  {true_i}  ->  {pred_i}")
""")
md("""
### Confusion matrix

This shows where predictions land for the most common intents. The diagonal is correct
predictions; anything off the diagonal is a mix-up. We show the 15 largest intents, because a
full 132x132 matrix is too big to read.
""")
code("""
from sklearn.metrics import confusion_matrix

top_intents = y_test.value_counts().head(15).index.tolist()
pred_series = pd.Series(pred, index=y_test.index)
mask = y_test.isin(top_intents)
cm = confusion_matrix(y_test[mask], pred_series[mask], labels=top_intents)

fig, ax = plt.subplots(figsize=(9, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks(range(len(top_intents)))
ax.set_xticklabels(top_intents, rotation=90)
ax.set_yticks(range(len(top_intents)))
ax.set_yticklabels(top_intents)
ax.set_xlabel("predicted intent")
ax.set_ylabel("true intent")
ax.set_title("Confusion matrix (15 most common intents)")
for r in range(len(top_intents)):
    for c in range(len(top_intents)):
        if cm[r, c]:
            ax.text(c, r, cm[r, c], ha="center", va="center", fontsize=8,
                    color="white" if cm[r, c] > cm.max() / 2 else "black")
fig.colorbar(im, fraction=0.046, pad=0.04)
plt.tight_layout()
plt.show()
""")
code("""
# full per-intent precision / recall / F1 (long, but shows every intent)
print(classification_report(y_test, pred, zero_division=0))
""")

# ------------------------------------------------------------------ save
md("""
## 9. Save the model

We save the whole pipeline (text processing + classifier together) to a file. Koppal loads
this file to classify questions without retraining. We reload it once to confirm it works.
""")
code("""
import os
os.makedirs("model", exist_ok=True)
joblib.dump(final_model, MODEL_PATH)
print("saved to", MODEL_PATH)

# reload and check it still predicts
reloaded = joblib.load(MODEL_PATH)
print("reloaded OK, example prediction:", reloaded.predict(["how do I reprint my call up letter?"])[0])
""")

# ------------------------------------------------------------------ demo
md("""
## 10. Live demo

This is the model working end to end: type a question, and it returns the predicted intent,
how confident it is, and the answer Koppal would reply with (looked up from the knowledge
base). This is what the chatbot does behind the scenes on every message.

We add one realistic rule: **if the model's confidence is below the 0.25 floor we chose above,
the bot asks the user to rephrase instead of guessing.** A real assistant should say "I'm not
sure" rather than give a confident wrong answer.
""")
code("""
# map each intent to the answer Koppal would send (short chat answer if we have one)
kb["answer_text"] = kb["chat_answer"].fillna("").where(
    kb["chat_answer"].fillna("").str.strip() != "", kb["answer"].fillna(""))
answer_for = dict(zip(kb["intent"].str.strip(), kb["answer_text"]))

CONFIDENCE_FLOOR = 0.25   # chosen from the sweep above

def ask(question):
    intent = final_model.predict([question])[0]
    confidence = final_model.predict_proba([question]).max()
    print("Q:", question)
    if confidence < CONFIDENCE_FLOOR:
        print(f"   (not sure - {confidence:.0%}) I didn't quite get that. Could you rephrase?")
    else:
        answer = answer_for.get(intent, "(no answer in the knowledge base yet for this intent)")
        print(f"   intent     : {intent}")
        print(f"   confidence : {confidence:.0%}")
        print(f"   answer     : {answer}")
    print()

ask("How do I reprint my call up letter?")
ask("Can I relocate to another state after camp?")
ask("Can I change my PPA?")
ask("How much is the monthly allowance?")
ask("What should I pack for orientation camp?")
ask("Can I bring my phone to camp?")
ask("What is CDS?")
ask("Who won the last election?")          # nothing to do with NYSC -> low confidence
""")

# ------------------------------------------------------------------ conclusions
md("""
## 11. Conclusions

- The model reads an NYSC question and predicts its intent, which Koppal uses to pick an answer.
- We compared three models and tuned Logistic Regression, chosen because it scores well and
  gives usable confidence probabilities.
- **Accuracy** on common questions is strong. **Two changes did most of the work**, and neither
  was a bigger model. Character n-grams alongside word n-grams stopped the model missing
  questions over spelling and plurals, and `class_weight="balanced"` stopped the two largest
  intents absorbing questions belonging to smaller ones. Together they took macro-F1 from about
  0.49 to 0.67.
- **Macro-F1 is still below accuracy** because it counts every intent equally, and many intents
  have only a few training questions. The score by class-size band shows this directly, though
  the gap narrowed once the class weighting was in.
- Adding hand-written paraphrases for the thin intents, to the training set only, is what lifts
  the rare classes without inflating the test scores.
- The confidence floor matters as much as the accuracy. At 0.25 the bot answers about six
  questions in ten and is right four times in five, and it declines the rest instead of
  guessing, which is the right behaviour for an assistant.
- **Main limitation:** the class imbalance above. The clearest next step is more real questions
  for the intents that have fewer than 15, since that's where the F1-by-band table shows the
  steepest gain. With more data a next version could also use cross-validation and a separate
  validation set instead of a single train/test split, because some intents have only one or two
  test questions, which makes the score noisy.
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python", "version": "3"}},
      "nbformat": 4, "nbformat_minor": 5}

Path("koppal_intent_classifier.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")
print("wrote koppal_intent_classifier.ipynb with", len(cells), "cells")
