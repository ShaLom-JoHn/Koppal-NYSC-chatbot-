# NYSC Chatbot — Project Journal

Decisions, reasoning, mistakes, and fixes across the project, in the order they happened.
Not a status report (see `PROJECT_STATUS.md` for that) — this is the *why* behind it.

Citations are marked `REF-###`, numbered sequentially across the whole journal, referenced
inline. Large mechanical work is logged as short batch entries (what was done, why, result)
rather than the full narrative format.

---

## Entry 1: Origin — data collection through v2 architecture

No dataset like this existed, so I started by collecting raw questions with no structure,
just scraping questions.

While collecting, I noticed similar questions and paraphrased duplicates. This is where the
idea of intents came from. I organized the scraped questions by intent to make sourcing
easier. Some of that grouping was done by matching words instead of actual context, which
was a mistake.

> **`REF-001` — Same word, different intent**
> *"What is a proper upkeep allowance for a young man (23 year old)?"*
> Grouped under `allowance_payment_issue` on the word "allowance." Actually a general
> finance question, not NYSC-specific. Moved to `general_meta_about_nysc`.

After questions were organized by intent, sourcing began, this time for answers. I realized
some questions couldn't get one straight answer. They needed a follow-up and a trigger
first. This led to building the structure and logic for how the conversation should branch.
Alongside this, I realized answers themselves needed a type: Statement, Procedural, or
Conditional, depending on whether a follow-up was needed at all.

With that logic defined, sourcing continued. New intents and questions, along with their
answers, were filled in following the now-defined structure. I didn't cross-check whether
every question grouped under an intent actually had an answer. Many didn't. The final
answer per intent covered the direct question but left out others grouped under the same
intent.

> **`REF-002` — Same intent, different (wrongly shared) answer**
> `change_local_government` and `clearance_ppa_wont_sign` were both answered with the
> `monthly_clearance` text. Wrong for both — fixed to answer what was actually asked
> (LGA transfer, supervisor refusing to sign).

This gap is what led to the audit (see Entry 2).

Before the audit's findings were fully worked through, I sketched what a real conversation
should look like structurally.

> **`REF-003` — Conversation logic in the raw dataset**
> Each row holds `answer_type`, `follow_up_trigger`, `follow_up_answer`. `answer_type`
> marks Statement or Conditional. The trigger and follow-up columns hold one flat branch
> per row — no distinction yet between a yes/no question, a multi-option branch, or a
> free-text answer.

I brought in AI to help build on that sketch: defined the problem, what I already had, my
skill level, and asked for simplicity, building on the existing work instead of rewriting
it. This surfaced changes, errors, and gaps in the original sketch.

Once the structure got overwhelming to reason through alone, I researched existing chatbot
frameworks. Found Rasa and Google Dialogflow. Checked their structure and why each feature
existed, borrowed the general idea, and simplified it down to what this project actually
needed. This became v2.

> **`REF-004` — Answer structure vs. architecture**
> The CSV stores one flat trigger and one flat follow-up answer per Conditional intent.
> v2 breaks this into `expected_type` (yes/no, branch-select, state, free text), and for
> intents asking two things at once, a separate `COMPOUND_SLOTS` structure tracking each
> sub-slot independently.

v2 also replaced assumption with data: v1's slot examples (`graduated`, `graduation_year`,
`institution`) were written before the real dataset existed. Auditing the actual 51
Conditional intents showed the real distribution — about 12 yes/no, about 35 branch-select,
1 free-text, 1 real state-name slot. Institution names and graduation years appear in
**zero** real follow-ups, so they were cut, not deferred.

---

## Entry 2: The audit

Triggered by the coverage gap found while sourcing per intent (Entry 1). Once triggered, the
audit didn't just fill gaps, it caught real errors already sitting in "finished" answers.

> **`REF-005` — A factual error, not just a gap**
> `state_of_deployment_choice` originally stated outright that no state-preference option
> exists during registration. 5+ independent current sources confirmed one does (selecting
> several preferred states, non-binding). Corrected.

> **`REF-006` — The self-containment rule**
> An early fix for a fingerprint-mismatch scenario was written as "see the `biometric_capture`
> answer" inside a different intent's follow-up — a dead pointer unless the bot is built to
> chase cross-references, which can't be assumed. Every answer was made self-contained from
> this point on; shared facts get copied into each intent that needs them.

The audit also caught structural mislabeling: 7 intents marked `Procedural` when they
actually had follow-up branches, meaning a bot reading the type field would have skipped
the follow-up for those. Relabeled `Conditional`.

The audit ran in a risk-ranked order after this (highest-hedged-confidence and
highest-question-count first), across four passes, closing every flagged item with either a
resolution or explicit, honest guidance where no source existed. Full detail lives in
`PROJECT_STATUS.md`'s appendix — not repeated here since it's already logged accurately
there.

---

## Entry 3: Building the KB and code on top of v2

With v2 settled, work moved from architecture back into content, starting with locations.
`logistics_travel_to_camp` got real data for all 37 camps, and in the process Kebbi's
address was corrected to Basaura/Jega LGA, overriding a stale third-party source that had
been sitting uncaught. State secretariat and NYSC headquarters locations were added the same
way. The LGA-inspector office question stayed procedural guidance rather than a table, since
774 LGAs isn't something worth trying to enumerate.

Course-to-PPA matching came next, and it needed an honest caveat rather than a clean answer:
70%+ of corps members end up in schools regardless of course, since course-based buckets are
a tendency, not a formula. A run of smaller standalone intents followed the same pattern of
sourcing and honest-caveat writing, and a full sweep of the Ground Zero guide turned up
mostly duplicate content, but three real gaps: a payment-scam warning, a consequence of the
senate list never being uploaded, and a new intent entirely, `siwes_vs_nysc`.

`chitchat.py` came from merging two separate things rather than building fresh: my own
researched version with 30+ categories and a time-of-day greeting, and an earlier version's
Nigerian pidgin keyword layer plus its stress-flagged empathy handling for sensitive intents.
Both were kept, nothing overwritten.

`slots.py` and `dialogue_manager.py` were where v2's decision flow actually got implemented,
branch 0 through 3, plus the compound sub-slot handling for the 5 intents that ask two
things in one question. Some gaps were left open on purpose rather than fixed blind: the
context stack has no depth cap yet, there's no explicit step classifying a message as
DETOUR versus a new topic, and the pending-slot check runs before the chitchat check, which
wastes a match attempt on chitchat sent mid-flow.

> **`REF-007` — A real bug in `term_lookup`, not just an untested path**
> `dialogue_manager.py` had zero references to the glossary dictionary despite the KB
> describing `term_lookup` as extraction-plus-lookup, not a branching Conditional. Any
> correctly classified `term_lookup` message would have returned the entire glossary and
> institution-code dump as one answer, not the specific term asked about. Fixed, and
> "Corper" was found missing from the glossary entirely during the same check — added.

Sourcing 12 new NYSC terms for the glossary caught one more real correction along the way.

> **`REF-008` — A guessed definition corrected against sources**
> PV had been recorded as "Personal Vetting form" with no source behind it. Two independent
> sources confirmed it actually means Payment Voucher. Corrected, and two lower-confidence
> terms (Otondo, CLO) were added with the disagreement stated openly in their definitions
> rather than picked silently.

What's left from here carries over from `PROJECT_STATUS.md` rather than repeating it:
`training_examples.csv` still needs the remaining 122 intents' examples, the classifier
still needs training (mine to do, not Claude's), and `nlu.py`, `main.py`, and
`dialogue_manager.py` are all built but have never actually been run.
