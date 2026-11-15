# Diabetes Risk Assistant

An end-to-end intelligent application that combines a **trained diabetes-risk classifier**
with an **LLM-powered natural-language interface**, tracked with **MLflow**, covered by a
**pytest** suite, configured from **YAML**, and packaged with **Docker**.

> "I'm a 35-year-old woman with a glucose level of 148 and a BMI of 33.6. What does your model predict?"
>
> *"Based on the information you provided, the model estimates a **75% probability** of diabetes
> (elevated risk). Your glucose of 148 mg/dL is the strongest driver, raising the estimate by about
> 15 points versus a typical value of 117 ... This is a statistical screening estimate, not a diagnosis."*

---

## 1. Project description

**What it does.** A user describes their health measurements in plain English. The system
extracts the model's input features from the text, runs a trained classifier, and replies
with the predicted probability of type 2 diabetes, the factors that drove it, and the caveats
that apply.

**Who it is for.** Health-education and screening contexts: someone who has a few numbers
from a recent check-up and wants to understand, in conversational terms, what a statistical
model makes of them before talking to a clinician. It is explicitly **not** a diagnostic tool.

**What problem it solves.** Tabular ML models are hard for non-experts to use: they need
exact feature names, units, and complete rows. The LLM layer removes that friction (it
converts mmol/L to mg/dL, computes BMI from height and weight, takes the diastolic value from
"130/85"), explains the output in context, and, crucially, *refuses to guess*: if a required
value is missing or implausible, it asks instead of producing a garbage prediction.

**Dataset.** [Pima Indians Diabetes](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)
(UCI / Kaggle): 768 women of Pima heritage, 8 clinical features, binary `Outcome`
(35 % positive). Downloaded by `scripts/get_data.py`; never committed.

| Feature | Meaning | Required by the assistant? |
|---|---|---|
| `Glucose` | 2-hour plasma glucose (mg/dL) | **yes** |
| `BMI` | body mass index (kg/m²), or height + weight | **yes** |
| `Age` | years | **yes** |
| `BloodPressure` | diastolic blood pressure (mm Hg) | optional |
| `Insulin` | 2-hour serum insulin (mu U/mL) | optional |
| `SkinThickness` | triceps skin-fold (mm) | optional |
| `Pregnancies` | count | optional |
| `DiabetesPedigreeFunction` | family-history score | optional |

Optional features that the user does not mention are filled with the training-set median and
the response says so.

## 2. Setup

Requires Python 3.12 (3.10+ should work; versions in `requirements.txt` are pinned for 3.12).

```bash
git clone <your-fork-url> diabetes-risk-assistant
cd diabetes-risk-assistant
python3.12 -m venv .venv && source .venv/bin/activate      # or: uv venv --python 3.12 .venv
pip install -r requirements.txt
```

**API key.** The LLM layer uses [Nebius AI Studio](https://studio.nebius.com/) through its
OpenAI-compatible API. Keys are read from environment variables only; nothing is hardcoded.

```bash
cp .env.example .env     # then paste your key into LLM_API_KEY
```

`LLM_BASE_URL` and `LLM_MODEL` can be changed to point at OpenAI or any other
OpenAI-compatible provider. If no key is set, the app runs in **offline mode**: a rule-based
parser and templated explanations replace the LLM, so the model is still usable and the test
suite never needs network access.

**Data.**

```bash
python scripts/get_data.py          # -> data/diabetes.csv (768 rows)
```

## 3. Usage

**Train every configuration and log to MLflow** (about 25 s on a laptop):

```bash
python -m src.train --all                   # 7 runs defined in configs/config.yaml
python -m src.train --experiment rf_default # or a single one
```

**Compare runs and export the best pipeline** (uses `mlflow.search_runs()`):

```bash
python scripts/compare_runs.py              # -> models/best_model.joblib + models/best_run.json
mlflow ui --backend-store-uri sqlite:///mlflow.db   # optional: browse runs at http://localhost:5000
```

**Run the assistant.**

```bash
streamlit run src/app.py                    # web chat UI (recommended)
python -m src.cli                           # interactive terminal loop
python -m src.cli "I'm 45, glucose 130, BMI 29, BP 120/82"
```

**Run the tests** (34 tests, ~1 s, no network needed):

```bash
pytest tests/ -v
```

**Scripted demo** (writes `docs/demo_transcript.md`, see §7):

```bash
python scripts/demo.py
```

**Docker** (optional): the image downloads the data, trains all runs, picks the best, and
serves the UI.

```bash
docker build -t diabetes-risk-assistant .
docker run -p 8501:8501 --env-file .env diabetes-risk-assistant   # omit --env-file for offline mode
```

## 4. Architecture

```
            user text                                     answer
                │                                            ▲
                ▼                                            │
┌───────────────────────────┐    JSON features    ┌──────────┴──────────────┐
│ 1. PARSE (LLM)            │ ─────────────────▶  │ 4. EXPLAIN (LLM)        │
│  intent + 8 features      │                     │  grounded in step-3     │
│  unit conversion, BMI     │                     │  numbers, with caveats  │
└───────────────────────────┘                     └──────────▲──────────────┘
                │ validate ranges / required                 │ probability, risk band,
                ▼                                            │ per-feature contributions
┌───────────────────────────┐                     ┌──────────┴──────────────┐
│ 2. GATE                   │  complete?  ──────▶ │ 3. PREDICT              │
│  missing → ask            │                     │  models/best_model.     │
│  implausible → ask        │                     │  joblib = Preprocessor  │
│  off-topic → decline      │                     │  + RandomForest         │
└───────────────────────────┘                     └─────────────────────────┘
```

* **`src/preprocess.py`** – `DiabetesPreprocessor`, a scikit-learn transformer: impossible
  zeros → NaN → median imputation, ordinal `BMICategoryCode` feature, standard scaling.
  All statistics are fit on the training split only (no leakage). It is the first step of the
  saved pipeline, so the app feeds *raw* user values and the model sees exactly what it saw in
  training.
* **`src/models.py` / `src/train.py`** – model factory + MLflow training loop. Every run logs
  the model type, every hyperparameter, preprocessing settings, split sizes, dataset
  description and SHA-256, five metrics, feature importances, and the full pipeline artifact.
* **`scripts/compare_runs.py`** – queries MLflow with `mlflow.search_runs()`, ranks by
  ROC-AUC then F1, and exports the winner to `models/best_model.joblib`.
* **`src/interface.py`** – the orchestration above. `parse_input` tries the LLM and falls back
  to a regex parser; `validate_features` range-checks; `predict` loads the pipeline and
  computes **what-if contributions** (how much the probability changes when each user value
  is swapped for the training median), a model-agnostic local explanation; `llm_response`
  hands those numbers to the LLM with instructions to use *only* them.
* **`src/llm.py`** – a small OpenAI-compatible client (Nebius by default) configured from
  env vars; returns `None` when no key is set, which switches the whole stack to offline mode.
* **`src/app.py`** (Streamlit) and **`src/cli.py`** – two thin front-ends over the same
  `DiabetesAssistant` class. The web UI exposes the parsed features, missing/invalid fields,
  the probability and a contribution chart under "How this answer was produced".

## 5. Results

Seven configurations were trained on an 80/20 stratified split (614 train / 154 test,
`random_state=42`) and evaluated on the held-out set. Full leaderboard from
`python scripts/compare_runs.py`:

| rank | run | model | ROC-AUC | F1 | accuracy | precision | recall |
|---|---|---|---|---|---|---|---|
| 1 | **rf_regularized** | random forest (500 trees, depth 6, min_leaf 4, balanced) | **0.823** | **0.683** | 0.747 | 0.609 | **0.778** |
| 2 | xgb_tuned | XGBoost (400 trees, lr 0.03, depth 3, subsampling, λ=2) | 0.823 | 0.628 | **0.753** | 0.667 | 0.593 |
| 3 | logreg_balanced | logistic regression (C=0.5, balanced) | 0.821 | 0.672 | 0.740 | 0.603 | 0.759 |
| 4 | logreg_baseline | logistic regression (C=1) | 0.820 | 0.540 | 0.701 | 0.587 | 0.500 |
| 5 | gb_default | gradient boosting (200 trees, lr 0.05) | 0.819 | 0.621 | 0.747 | 0.653 | 0.593 |
| 6 | rf_default | random forest (300 trees, unlimited depth) | 0.814 | 0.621 | 0.747 | 0.653 | 0.593 |
| 7 | mlp | MLP (32-16, early stopping) | 0.810 | 0.562 | 0.747 | **0.714** | 0.463 |

**Why `rf_regularized`.** ROC-AUC is the primary selection metric because it is
threshold-independent and robust to the 65/35 class imbalance; F1 breaks ties. The
regularized forest is best on both, and its recall of 0.78 is the highest of the seven. For a
*screening* assistant, missing a diabetic patient (false negative) is the costlier error, so
trading some precision for recall is the right call. `xgb_tuned` is statistically
indistinguishable on AUC but catches 19 % fewer positives.

**Findings.**
* The published ceiling for this dataset is roughly 0.82–0.85 AUC; 0.823 on a 154-row test
  set is in that range. Several very different model families land within 0.01 AUC of each
  other, which says the signal is limited by the data (8 noisy features, 768 rows), not the
  algorithm.
* `class_weight="balanced"` mattered more than the algorithm choice: it lifted the logistic
  regression's recall from 0.50 to 0.76 and the forest's from 0.59 to 0.78 at almost no AUC
  cost.
* Treating zeros as missing was essential: 49 % of `Insulin` and 30 % of `SkinThickness`
  values are zero placeholders. Leaving them in gives tree models a spurious "zero means
  healthy" split.
* Glucose dominates feature importance (~30 %), followed by BMI, age and the pedigree
  function, which matches clinical intuition and makes the LLM's explanations sensible.

## 6. Repository structure

```
.
├── README.md
├── requirements.txt          # pinned
├── Dockerfile                # trains + serves the Streamlit app
├── .env.example              # LLM_API_KEY / LLM_BASE_URL / LLM_MODEL template
├── .gitignore                # excludes data/, models/, mlruns/, mlflow.db, .env
├── pytest.ini
├── configs/config.yaml       # data, preprocessing, MLflow and all 7 experiment configs
├── src/
│   ├── preprocess.py         # cleaning, imputation, BMI category, scaling (leakage-safe)
│   ├── models.py             # model factory
│   ├── train.py              # MLflow-tracked training (reads config.yaml)
│   ├── evaluate.py           # metrics, confusion counts, what-if contributions
│   ├── llm.py                # OpenAI-compatible client (Nebius default), env-var config
│   ├── interface.py          # parse → validate → predict → explain
│   ├── app.py                # Streamlit chat UI
│   └── cli.py                # terminal interface
├── scripts/
│   ├── get_data.py           # download dataset
│   ├── compare_runs.py       # mlflow.search_runs() leaderboard + best-model export
│   └── demo.py               # scripted demo → docs/demo_transcript.md
├── tests/
│   ├── conftest.py
│   ├── test_preprocess.py    # 10 tests: missing values, encoding, scaling, immutability, leakage
│   ├── test_model.py         # 5 tests: shape/type, performance floor, sanity ordering
│   └── test_interface.py     # 13 tests: parsing accuracy, LLM JSON, edge cases, grounding
├── docs/demo_transcript.md
├── data/.gitkeep             # data downloaded, not committed
└── models/.gitkeep           # artifacts exported from MLflow, not committed
```

## 7. Demo

`docs/demo_transcript.md` is a scripted walkthrough produced by `python scripts/demo.py`.
It shows a complete query, a query needing unit conversion (`5'4" and 150 lbs`, `130/85`),
a low-risk profile, and four edge cases: a missing required feature (the assistant asks for
glucose), an implausible value (`glucose 5000` is rejected), an out-of-scope question (the
weather) and a question about the tool itself. The same examples are one click away in the
Streamlit sidebar for a live demo.

## 8. Reflection

**What I learned.** The hardest part of an "LLM over a model" system is not the LLM call,
it is the *contract* between the two halves. Making the LLM emit a strict JSON schema with
`null` for anything not stated, and then range-checking that JSON in deterministic Python,
turned "the LLM hallucinated a glucose value" from a correctness bug into something that
cannot happen by construction. Likewise, generating the explanation from the model's own
numbers (probability + what-if contributions) and telling the LLM to use *only* those keeps
the response grounded. Having the entire path except the network call be deterministic also
made it testable: 34 tests run in a second with a fake LLM.

**What was challenging.**
* MLflow 3.x retired the plain file store and switched sklearn serialization to `skops`,
  which rejects custom transformer classes; moving to SQLite tracking and cloudpickle
  serialization fixed both.
* Deciding what "incomplete" means. Requiring all eight Pima features would make the tool
  useless (nobody knows their 2-hour serum insulin). Requiring glucose, BMI and age and
  filling the rest with medians, while saying so in the answer, was the compromise.
* The dataset itself: zeros as missing values, a small test set (154 rows) where one
  prediction moves accuracy by 0.65 %, and a population (Pima women) that limits how far the
  model generalizes. The caveats in every response are there for a reason.

**What I would improve with more time.**
* Calibrate probabilities (Platt / isotonic) and report Brier score; a "75 %" should mean 75 %.
* Use repeated stratified cross-validation for model selection instead of a single split, and
  log the CV distribution to MLflow.
* Register the winning model in the MLflow Model Registry and have the app load by stage
  (`models:/diabetes-risk-classifier/Production`) instead of a joblib file.
* Multi-turn memory in the UI, so "my glucose is 130" after a clarifying question merges with
  the previous features rather than starting over.
* Evaluate the LLM parser itself on a labelled set of paraphrased inputs and track its
  extraction accuracy as a metric alongside the model's.

---

*Not medical advice. The model is trained on a small historical dataset of Pima women and
produces statistical screening estimates only.*
