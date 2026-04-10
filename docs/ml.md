# How Mail Butler classifies emails

This document explains the machine learning and classification pipeline used in
Mail Butler. It is written for developers who may not have a data science
background but want to understand what happens under the hood.

## The big picture

When you run `mail-butler classify`, each email goes through up to **3 layers**
of classification, from cheapest to most expensive:

```
Email ──> Rules ──> sklearn (ML) ──> Ollama LLM
           │            │                 │
           ▼            ▼                 ▼
        Confident?   Confident?      Best guess
         YES → done   YES → done       → done
         NO  → next   NO  → next
```

This is called a **classification cascade**. The idea is simple: don't use a
slow, expensive tool when a fast, cheap one already gives a good answer.

Each classifier returns two things:
- A **category** (e.g. `newsletter`, `promotion`, `transactional`)
- A **confidence score** between 0.0 and 1.0

If the confidence is above the threshold (default: 0.7), we accept the result.
Otherwise, we escalate to the next classifier.

> Source: `src/mail_butler/classifier/__init__.py` — the `auto_classify` function

---

## Layer 1: Rule-based classifier

**Speed:** instant (microseconds per email)
**How it works:** simple if/else logic on email metadata

This is not machine learning at all — it is a set of hand-written rules that
check email headers and subject patterns. Think of it as a decision tree that a
human wrote by hand.

### What it looks at

| Signal | Where it comes from | What it means |
|--------|-------------------|---------------|
| `List-Unsubscribe` header | RFC 2369 standard header | Email is from a mailing list |
| `List-ID` header | RFC 2919 standard header | Email belongs to a named list |
| `Precedence: bulk` | Email header | Mass-sent email |
| `X-Mailer` header | Email header | Which software sent the email |
| Sender address | `From:` header | `noreply@` = automated, known domains = transactional |
| Subject line | `Subject:` header | Pattern matching for keywords |

### How the rules work (simplified)

```
IF sender is "noreply@" AND subject contains "order" or "receipt"
  → TRANSACTIONAL (confidence: 0.9)

IF List-Unsubscribe header exists AND subject contains "% off" or "sale"
  → PROMOTION (confidence: 0.9)

IF List-ID header exists AND subject does NOT look promotional
  → NEWSLETTER (confidence: 0.8)

IF sender domain is linkedin.com AND subject contains "followed you"
  → SOCIAL (confidence: 0.85)

IF nothing matches
  → UNKNOWN (confidence: 0.0) → escalate to next classifier
```

The real implementation has 11 rules, ordered from most specific to most
general. Each rule is tested in order, and the first one that matches wins.

### Why start with rules?

- **Speed**: rules process thousands of emails per second
- **Transparency**: you can read the code and understand exactly why an email
  was classified a certain way
- **No training needed**: works from day one, even with zero historical data
- **High precision on clear cases**: an email from `noreply@paypal.com` with
  subject "Your receipt" is always transactional — no ML needed

### Limitations

Rules fail on ambiguous emails. An email from `john@company.com` with subject
"Quick update" could be personal, work, or a newsletter. Rules will return
`UNKNOWN` for these cases.

> Source: `src/mail_butler/classifier/rules.py`

---

## Layer 2: sklearn (traditional machine learning)

**Speed:** fast (milliseconds per email, seconds for a batch of thousands)
**How it works:** learns patterns from previously classified emails

This is where actual machine learning comes in. We use a classic technique
called **TF-IDF + Naive Bayes** — a well-proven approach for text
classification that has been used for spam filters since the early 2000s.

### Step-by-step: how text becomes a prediction

#### 1. Feature extraction: turning text into numbers

Machine learning algorithms cannot read text. They work with numbers. So we
need to convert each email into a list of numbers (called a **feature vector**).

We use **TF-IDF** (Term Frequency - Inverse Document Frequency):

```
Input:   "Your order has shipped from Amazon"
Output:  [0.0, 0.0, 0.42, 0.0, 0.31, ..., 0.0]   (up to 10,000 numbers)
```

Each number represents how important a specific word (or pair of words) is
for this email, relative to all other emails.

**TF (Term Frequency):** How often does this word appear in THIS email?
A word that appears 5 times is probably more important than one that appears
once.

**IDF (Inverse Document Frequency):** How rare is this word across ALL emails?
The word "the" appears everywhere, so it gets a low score. The word "invoice"
is rarer and more meaningful, so it gets a high score.

**TF-IDF = TF × IDF**: a word is important if it appears often in this email
but rarely in other emails.

```
"the"     → TF: high (appears often)  × IDF: low  (appears everywhere)  = low score
"invoice" → TF: medium               × IDF: high (specific to some emails) = high score
```

The result is a **sparse vector** — most values are 0.0 because each email only
contains a tiny fraction of all possible words. This is normal and efficient.

**Configuration in our code:**
- `max_features=10000`: keep the 10,000 most informative words/word-pairs
- `ngram_range=(1, 2)`: consider single words ("order") AND word pairs
  ("order confirmed") — pairs capture meaning that single words miss
- `sublinear_tf=True`: use `1 + log(TF)` instead of raw count, so a word
  appearing 100 times is not scored 100x more than one appearing once
- `min_df=2`: ignore words that appear in only one email (probably noise)
- `max_df=0.95`: ignore words that appear in 95%+ of emails (too common)

#### 2. Building the text feature

Before feeding text to TF-IDF, we combine multiple email fields into one
string. The subject is included twice to give it more weight:

```python
text = subject + " " + subject + " " + body_preview + " " + sender_domain
```

Why repeat the subject? Because the subject is a dense summary of the email's
purpose. In a bag-of-words model like TF-IDF, repeating it is a simple way to
say "these words matter more."

#### 3. Classification: Multinomial Naive Bayes

Once we have numbers, we need an algorithm that learns which patterns of
numbers correspond to which categories. We use **Multinomial Naive Bayes**.

**The core idea (Bayes' theorem):**

```
P(category | words) = P(words | category) × P(category) / P(words)
```

In plain English: "What is the probability that this email is a NEWSLETTER,
given the words it contains?"

The algorithm answers this by looking at training data:
- Among all known newsletters, how often does the word "unsubscribe" appear?
- Among all known promotions, how often does the word "discount" appear?
- What fraction of all emails are newsletters vs. promotions?

**"Naive"** means the algorithm assumes each word is independent — it doesn't
understand that "free shipping" is a phrase, it just sees "free" and "shipping"
as two separate signals. This is obviously wrong, but it works surprisingly
well in practice because the statistical signal is strong enough.

**"Multinomial"** means it works with word counts (how many times each word
appears), which matches our TF-IDF output.

**Why Naive Bayes for email classification?**
- Very fast to train (seconds, not hours)
- Works well with small training sets (50-100 examples per category)
- Naturally outputs probabilities (our confidence score)
- Low memory footprint
- Hard to overfit (the "naive" assumption acts as a regularizer)

#### 4. Confidence score

Naive Bayes outputs a probability for each category. We pick the highest one
as the prediction, and use its value as the confidence score:

```
P(newsletter)    = 0.82  ← highest → prediction: newsletter, confidence: 0.82
P(promotion)     = 0.11
P(transactional) = 0.04
P(personal)      = 0.02
P(spam)          = 0.01
```

If the top probability is below the threshold (0.7), we don't trust the
prediction and escalate to the LLM.

### How training works (bootstrapping)

A ML model needs labeled training data — emails where we already know the
correct category. But when you first install Mail Butler, there is no labeled
data.

We solve this with **bootstrapping from rules**:

```
1. Run the rule-based classifier on all emails
2. Keep only high-confidence results (confidence >= 0.7)
3. Use those as training data for sklearn
4. Save the trained model to disk ({account-name}.joblib)
```

This means the sklearn model starts as an "accelerated version" of the rules:
it learns the same patterns but can generalize to cases the rules miss. Over
time, as you classify more emails, the model improves.

**Each account has its own model** because different mailboxes have different
patterns — your personal Gmail and your work Outlook will have very different
senders and vocabulary.

### The sklearn pipeline

In scikit-learn, a **Pipeline** chains multiple processing steps together:

```
Raw text → TfidfVectorizer → MultinomialNB → Category
              (step 1)          (step 2)
```

The pipeline ensures that the same transformations are applied during training
and prediction. Without it, you might accidentally process text differently
when classifying new emails vs. when training, which would break the model.

> Source: `src/mail_butler/classifier/sklearn_classifier.py`

---

## Layer 3: LLM classifier (Ollama)

**Speed:** slow (1-5 seconds per email)
**How it works:** asks a language model to reason about the email

This is the fallback for emails that neither rules nor sklearn can classify
with confidence. We send the email to a local LLM running on Ollama.

### How it works

We send a structured prompt to the LLM via Ollama's HTTP API:

```
System: You are an email classifier. Classify the email into exactly one
        category: newsletter, promotion, personal, work, transactional,
        spam, social, notification.
        Respond with ONLY a JSON object: {"category": "...", "confidence": 0.0-1.0}

User:   From: john.smith@company.com
        Subject: Q3 budget review - action needed
        Preview: Hi team, please review the attached budget proposal...
```

The LLM responds with:
```json
{"category": "work", "confidence": 0.92}
```

### Why a local LLM?

- **Privacy**: emails never leave your machine — no API calls to OpenAI or
  anyone else
- **Nuance**: LLMs understand context that TF-IDF cannot. "Quick update" from
  a colleague is work; "Quick update" from a SaaS tool is notification
- **Zero training needed**: the model already understands language and email
  conventions from its pre-training data

### Why not use the LLM for everything?

- **Speed**: 1-5 seconds per email vs. microseconds for rules. For 10,000
  emails, that is 3-14 hours vs. seconds
- **Resource usage**: runs on GPU/CPU, significant memory footprint
- **Overkill for obvious cases**: you don't need a 4-billion parameter model to
  know that an email from `noreply@paypal.com` with "Your receipt" is
  transactional

### Temperature: 0.1

We set `temperature: 0.1` (very low). Temperature controls how "creative" the
LLM is:
- `temperature: 0.0` = always pick the most likely next word (deterministic)
- `temperature: 1.0` = sample from the full distribution (creative, varied)

For classification, we want consistency, not creativity. The same email should
get the same category every time.

> Source: `src/mail_butler/classifier/llm_classifier.py`

---

## The auto strategy: putting it all together

When you run `mail-butler classify --method auto`, the `auto_classify` function
orchestrates the cascade:

```python
def auto_classify(email, rules, sklearn, llm, threshold=0.7):
    # 1. Try rules (free, instant)
    result = rules.classify(email)
    if result.category != UNKNOWN and result.confidence >= threshold:
        return result

    # 2. Try sklearn if model exists (fast, learned)
    if sklearn is not None:
        result = sklearn.classify(email)
        if result.confidence >= threshold:
            return result

    # 3. Fall back to LLM (slow, smart)
    if llm is not None:
        result = llm.classify(email)
        if result.category != UNKNOWN:
            return result

    # 4. Give up — return whatever rules said
    return rules.classify(email)
```

In practice, the distribution looks roughly like:
- **60-70%** of emails are caught by rules (obvious newsletters, promotions,
  transactional)
- **20-30%** are caught by sklearn (learned patterns)
- **5-10%** need the LLM (ambiguous emails)
- **<1%** remain UNKNOWN

> Source: `src/mail_butler/classifier/__init__.py`

---

## Glossary

| Term | Meaning |
|------|---------|
| **TF-IDF** | Term Frequency - Inverse Document Frequency. A way to turn text into numbers by measuring word importance. |
| **Naive Bayes** | A probabilistic classifier that assumes features are independent. Fast and effective for text. |
| **Feature vector** | A list of numbers representing one data point (one email). |
| **Sparse vector** | A feature vector where most values are zero. Efficient to store and compute. |
| **Pipeline** | A chain of processing steps (transform → classify) that stays consistent between training and prediction. |
| **Bootstrapping** | Training a model using the output of a simpler system (rules) as labeled data. |
| **Confidence threshold** | The minimum probability required to trust a classification (default: 0.7 = 70%). |
| **Temperature** | LLM parameter controlling randomness. Low = deterministic, high = creative. |
| **Cascade** | Running classifiers in order of cost, stopping at the first confident answer. |
| **Overfitting** | When a model memorizes training data instead of learning general patterns. Naive Bayes resists this. |
| **joblib** | Python library for saving/loading sklearn models to disk. |
