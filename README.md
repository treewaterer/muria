---
license: apache-2.0
base_model: google/gemma-4-12B-it
library_name: transformers
pipeline_tag: text-generation
tags:
- decision-making
- multiple-choice
---

# muria

muria is a fine-tune of [google/gemma-4-12B-it](https://huggingface.co/google/gemma-4-12B-it) that answers multiple-choice
decision questions with a probability for each option. It includes a calibrated last-layer head (`vbll_head.safetensors`,
`vbll_head.json`) whose output is the answer. For ordinary multiple-choice questions that output is the softmax of the option
letters' logits at the answer position, divided by one fitted temperature (T = 0.970); no posterior is involved. For yes / no
questions, a Bayesian last-layer residual on the model's hidden state (a Laplace posterior, integrated over) adjusts the two
logits.

Two parts of the model are not optional:
- **Attention.** In the full-attention layers, each prompt's tokens attend to every token of the same prompt; the sliding-window
  layers stay causal (window 1,024). transformers' stock attention would run the full-attention layers causally and give different
  answers, so the model is served with the `jev_mem` attention function in `fullft_mem.py`. It registers with transformers'
  attention interface and runs on PyTorch SDPA (memory-efficient and flash kernels). The engine refuses any other attention.
- **The head.** The letter logits on their own are not the model's answer.

The weights (`model-*.safetensors`, `vbll_head.safetensors`) are hosted here on the Hugging Face Hub; the GitHub mirror,
[treewaterer/muria](https://github.com/treewaterer/muria), holds the code and this card only.

## Usage

The repository includes a [Decision Index](https://github.com/apolinario/decision-index) engine, `dikit_engine.py`, with
`garas_vbll_engine.py`, `vbll_core.py`, `fullft_mem.py`, `garas_engine.py` and `garas_core.py`. It needs the Decision Index kit:

```bash
pip install "git+https://github.com/apolinario/decision-index@9eb2dbe2a358004c8782c66e40a83ac07b953fec"
REPO=$(python -c "from huggingface_hub import snapshot_download; print(snapshot_download('plantwaterer/muria'))")
PYTHONPATH="$REPO" python -m decision_index run --engine dikit_engine:DikitEngine --option model="$REPO" --compact --out runs/muria
```

In Python:

```python
import sys
from huggingface_hub import snapshot_download

repo = snapshot_download("plantwaterer/muria")
sys.path.insert(0, repo)
from dikit_engine import DikitEngine          # needs the decision_index kit installed

engine = DikitEngine(model=repo)               # bf16 on "cuda"; the head runs on the CPU
response, _ = engine(
    "The forecast gives a 70% chance of heavy rain this afternoon. The match is outdoors, on grass.",
    {"plan": {"type": "choice", "instructions": "What should the organisers do?",
              "criteria": {"play": "Play as planned", "indoors": "Move it indoors", "postpone": "Postpone it"}},
     "rain": {"type": "noul", "instructions": "Is heavy rain likely?"}})
print(response["answers"])
```

The inference procedure is as follows:
- **Prompts:** one prompt per question, with the model's chat template and the options lettered A-Z a-z. A choice question's
  options are shown in one fixed non-identity order derived from a hash of the request. Yes / no questions use the options `Yes`
  and `No`; when the request gives their definitions (its `criteria`), they are shown as `Yes: <definition>` / `No: <definition>`.
- **Read-out:** one forward pass per prompt. The final hidden state and the option letters' logits (fp32, with the model's logit
  soft-cap) go through the head, on the CPU in float64, which gives the probabilities.
- **Batching:** the prompts of one request run in packed rows of up to 32,768 tokens, each prompt with its own positions, so a
  prompt only attends to itself.
- **More than 52 options:** handled in two rounds.
- **No truncation:** a request that does not fit 32,768 tokens is reported as unsupported.

Requirements: one GPU (the bf16 weights are 24 GB, and the packed rows add activations for up to 32,768 tokens). Tested on
1 x H100 80GB and 1 x RTX PRO 6000 (96 GB) with torch 2.14.1 (CUDA 13.0), transformers 5.12.1, huggingface_hub, safetensors and
numpy. bf16 results differ slightly between GPU types.

## Evaluation

**Decision Index 0.3, public panel (37 benchmarks).** The run used this repository's weights, head and engine, with the kit's runner
and scorer at commit 9eb2dbe2, in bf16. The full public suite ran as 8 row shards on single GPUs (shards 0, 1, 2 and 7 on H100 80GB,
3 to 6 on RTX PRO 6000), merged without changes. This is our own run of the public kit, not a leaderboard entry. The run files are at
[plantwaterer/muria-decision-index](https://huggingface.co/datasets/plantwaterer/muria-decision-index).

| | public index | knowledge | language | retrieval | tools | arts |
|---|---|---|---|---|---|---|
| board rule (the 3 test items listed below counted as wrong) | 51.40 | 31.41 | 56.62 | 58.90 | 70.82 | 39.06 |
| as scored | 51.40 | 31.41 | 56.62 | 58.90 | 70.82 | 39.06 |

140,620 requests were run and 140,178 scored after the official exclusions; all of them were answered (0 unsupported, 0 errors).
Nothing was truncated. The two rows are equal at two decimals: one of the 3 listed items was answered correctly, and counting it
as wrong moves ANLI's score from 0.7171 to 0.7168.

**Latency** (our approximation of the board's check, which uses a private sample): the kit runner's per-request wall time on the
four RTX PRO 6000 shards of the run above, one request at a time, 70,309 answered requests:

| median | mean | 80th percentile |
|---|---|---|
| 48 ms | 209 ms | 169 ms |

### Overlap with evaluation data

No test split of any benchmark was used to build the training data. The head's fit and calibration data include train-split rows
of CLINC150, GSM8K, Amazon ESCI, BANKING77 and WinoGrande, and units from DocNLI's training split, some of which contain
premise-hypothesis pairs from ANLI (see the note at the end of this card); the training data include Lichess puzzles (a different
dataset from ChessBench's).

An audit of the training text, the head's data included, against the Decision Index public suite found 3 ANLI test items that are
near-duplicates of ANLI pairs inside DocNLI units in the head's fit data: `12:ANLI:ANLI:test_r2:5460dddd-0280-47bd-9148-1422b6729a4d`,
`12:ANLI:ANLI:test_r2:f6a3b2f4-863e-41af-bde1-762da7822ebd` and `12:ANLI:ANLI:test_r3:d469aabc-dd2a-4bd7-b840-ab39764b6625`. They are
counted as wrong in the board-rule number above, as the board does for test items found in a model's training data (see the
Contamination section of the [Decision Index methodology](https://multimodalart-jev-decision-index.static.hf.space/methodology.html)).

## Licence

The weights and the head are released under Apache-2.0, and the base model, google/gemma-4-12B-it, is Apache-2.0. The training data
come from the sources below, under the licences and terms listed. Some of these sources carry use conditions that apply to the data
itself; they are repeated here: US Census PUMS (never use the data to identify any person or household), NVD (not endorsed or
certified), California DMHC (modified; not official government data), MITRE CVE / CWE (copyright notices below), XNLI (CC BY-NC
4.0, non-commercial; its translations are part of AmericasNLI), and, in the head's fit and calibration data only, ANLI (CC BY-NC
4.0, non-commercial; its premise-hypothesis pairs are part of DocNLI) and wikiHow (CC BY-NC-SA 3.0, non-commercial; via
goal-step-wikihow). The note at the end of this card gives the counts.

## Data attribution

muria was trained on decision questions built from the public data sources below, plus synthetic questions generated by the
author (no third-party data). Thanks to everyone who builds and maintains them. Records were reformatted, and many were modified
(fields selected, text clipped, identifiers masked), so none of the material should be read as an official copy of its source.

### Open data and public records

- **GDELT Project**: event data. Data from The GDELT Project, https://www.gdeltproject.org/.
- **Wikipedia**: article text from the English Wikipedia dump of 2023-11-01, via
  [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.en`, by Wikipedia contributors.
  Licensed CC BY-SA 3.0/4.0 and GFDL. https://en.wikipedia.org/
- **Wikipedia Articles for Deletion**: deletion debates by English Wikipedia contributors (CC BY-SA 3.0/4.0, GFDL), via the AfD
  corpus of Mayfield & Black (2019), https://github.com/emayfield/AFD_Decision_Corpus (the repository's code is GPL-3.0; the
  corpus files are distributed separately).
- **Wikidata**: structured data (CC0 1.0), and Requests-for-deletion discussions by the Wikidata contributors (CC BY-SA 4.0).
  https://www.wikidata.org/
- **Swiss Judgment Prediction XL**: [rcds/swiss_judgment_prediction_xl](https://huggingface.co/datasets/rcds/swiss_judgment_prediction_xl)
  (Rasiah et al., 2023), CC BY-SA 4.0. Decisions of the Swiss Federal Supreme Court, anonymised by the court.
- **NHTSA** Office of Defects Investigation investigations and complaints (public domain), https://static.nhtsa.gov/odi/ffdd/
- **OSHA** Severe Injury Reports and enforcement data, US Department of Labor (public domain), https://www.osha.gov/severe-injury-reports
- **NTSB** aviation accident database (public domain), https://data.ntsb.gov/avdata
  (Added 2026-10-22: the OSHA and NTSB records above were in this model's training and head data but missing from this list.)
- **openFDA** enforcement reports and device recalls, CC0 1.0, https://open.fda.gov/
- **ClinicalTrials.gov** (US National Library of Medicine), via the ClinicalTrials.gov API v2; data processed by ClinicalTrials.gov on
  2026-10-01. Modified: selected registration fields only, free text clipped, trial ids masked, study-status sentences removed. Some
  ClinicalTrials.gov data may be subject to third-party copyright outside the United States.
- **ClinVar** (NCBI): Landrum et al., Nucleic Acids Research 2018, PMID 29165669.
- **Orphanet** inheritance data, CC BY 4.0, https://www.orpha.net/
- **CIViC** (civicdb.org), CC0 1.0: Griffith et al., Nature Genetics 2017, PMID 28138153.
- **California Department of Managed Health Care**, "DMHC IMR Data, 2001 - Current", CC BY,
  https://data.ca.gov/dataset/independent-medical-review-imr-determinations-trend. Modified; not official government data.
- **NOAA / NWS Storm Prediction Center** text products (public domain), via the Iowa Environmental Mesonet, Iowa State University.
- **LIGO / Virgo / KAGRA** public alerts and the GWOSC event catalogue, CC BY 4.0, by the LIGO Scientific, Virgo and KAGRA
  Collaborations. This model uses data obtained from the Gravitational Wave Open Science Center (gwosc.org), a service of the LIGO
  Scientific Collaboration, the Virgo Collaboration and KAGRA.
- **USGS** PAGER and ComCat (public domain). Population exposure is from ORNL LandScan (CC BY 4.0).
- **National Vulnerability Database** (NIST, public domain) and **CVE** records (CVE Terms of Use; Copyright The MITRE Corporation).
  **CWE**: Copyright 2006-2026, The MITRE Corporation; CWE is a trademark of The MITRE Corporation. This product uses the NVD API
  but is not endorsed or certified by the NVD.
- **American Community Survey PUMS**, US Census Bureau (CC0 1.0), with task definitions from
  [folktables](https://github.com/socialfoundations/folktables) (Ding et al., 2021; MIT). Never to be used to identify any person or
  household.
- **OpenReview** reviews and scores, CC BY 4.0 (titles and abstracts: CC0 1.0), https://openreview.net/
- **Lichess** open database, CC0 1.0, https://database.lichess.org/
- **Amazon ESCI** shopping queries, [amazon-science/esci-data](https://github.com/amazon-science/esci-data) (Reddy et al., 2022),
  Apache-2.0. Copyright Amazon.com, Inc. or its affiliates.
- **SWE-rebench**: [nebius/SWE-rebench-openhands-trajectories](https://huggingface.co/datasets/nebius/SWE-rebench-openhands-trajectories)
  (Trofimova et al., 2025), CC BY 4.0.
- **Open-Jev**: [ZefanCai/Open-Jev](https://huggingface.co/datasets/ZefanCai/Open-Jev), CC0 1.0.
- **LLM Bayesian Teaching Data**: Qiu, L. (2025), Zenodo, doi:10.5281/zenodo.17677329, CC BY 4.0.

### Research datasets

| dataset | reference | licence |
|---|---|---|
| BANKING77 | Casanueva et al., 2020 | CC BY 4.0 |
| CLINC150 | Larson et al., 2019 | CC BY 3.0 |
| GoEmotions | Demszky et al., 2020 | Apache-2.0 |
| GSM8K | Cobbe et al., 2021 | MIT |
| HelpSteer2 | Wang et al., 2024 | CC BY 4.0 |
| Measuring Hate Speech | Kennedy et al., 2020 | CC BY 4.0 |
| IBM Debater ArgQ-Rank-30kArgs | Gretz et al., 2019 | CC BY 3.0 |
| BBQ | Parrish et al., 2022 | CC BY 4.0 |
| CaSiNo | Chawla et al., 2021 | CC BY 4.0 |
| Circa | Louis et al., 2020 | CC BY 4.0 (dataset card; its README links CC BY-SA 4.0) |
| CodeContests | Li et al., 2022 | CC BY 4.0 |
| CosmosQA | Huang et al., 2019 | CC BY 4.0 |
| CommonsenseQA | Talmor et al., 2019 | MIT |
| e-CARE | Du et al., 2022 | MIT |
| Social IQa | Sap et al., 2019 | CC BY 4.0 |
| WinoGrande | Sakaguchi et al., 2019 | CC BY (code Apache-2.0) |
| MedMCQA | Pal et al., 2022 | Apache-2.0 |
| PubMedQA | Jin et al., 2019 | MIT |
| Stanford Politeness Corpus, via ConvoKit | Danescu-Niculescu-Mizil et al., 2013 | CC BY 4.0 |
| Scruples | Lourie et al., 2021 | Apache-2.0 |
| TabFact | Chen et al., 2020 | CC BY 4.0 |
| QASPER | Dasigi et al., 2021 | CC BY 4.0 |

Through [tasksource](https://huggingface.co/tasksource) (Sileo, 2024), each dataset under its own licence as recorded on its
dataset page:

| dataset (Hugging Face) | licence |
|---|---|
| hope_edi/english | CC BY 4.0 |
| [AndyChiang/cloth](https://huggingface.co/datasets/AndyChiang/cloth) | MIT |
| [bogdanminko/wildguardmix-cleaned](https://huggingface.co/datasets/bogdanminko/wildguardmix-cleaned) | ODC-BY |
| [cambridgeltl/xcopa](https://huggingface.co/datasets/cambridgeltl/xcopa) | CC BY 4.0 |
| [copenlu/scientific-exaggeration-detection](https://huggingface.co/datasets/copenlu/scientific-exaggeration-detection) | GPL-3.0 |
| [corypaik/prost](https://huggingface.co/datasets/corypaik/prost) | Apache-2.0 |
| [declare-lab/cicero](https://huggingface.co/datasets/declare-lab/cicero) | MIT |
| [dynabench/dynasent](https://huggingface.co/datasets/dynabench/dynasent) | CC BY 4.0 |
| [fancyzhx/amazon_polarity](https://huggingface.co/datasets/fancyzhx/amazon_polarity) | Apache-2.0 |
| [flaitenberger/wnut_17](https://huggingface.co/datasets/flaitenberger/wnut_17) | CC BY 4.0 |
| [GGLab/PARADISE](https://huggingface.co/datasets/GGLab/PARADISE) | MIT |
| [google-research-datasets/poem_sentiment](https://huggingface.co/datasets/google-research-datasets/poem_sentiment) | CC BY 4.0 |
| [google/civil_comments](https://huggingface.co/datasets/google/civil_comments) | CC0 1.0 |
| [jana4/turingbench-humanized](https://huggingface.co/datasets/jana4/turingbench-humanized) | Apache-2.0 |
| [joey234/nan-nli](https://huggingface.co/datasets/joey234/nan-nli) | CC BY-SA 4.0 |
| [juletxara/xstory_cloze](https://huggingface.co/datasets/juletxara/xstory_cloze) | CC BY-SA 4.0 (translations of the StoryCloze test) |
| [lasha-nlp/CONDAQA](https://huggingface.co/datasets/lasha-nlp/CONDAQA) | Apache-2.0 |
| [mteb/NusaX-senti](https://huggingface.co/datasets/mteb/NusaX-senti) | CC BY-SA 4.0 |
| [mteb/tweet_sentiment_multilingual](https://huggingface.co/datasets/mteb/tweet_sentiment_multilingual) | CC BY 3.0 |
| [nala-cub/americas_nli](https://huggingface.co/datasets/nala-cub/americas_nli) | CC BY-SA 4.0 (a translation of a subset of XNLI, whose licence is CC BY-NC 4.0) |
| [nedjmaou/MLMA_hate_speech](https://huggingface.co/datasets/nedjmaou/MLMA_hate_speech) | MIT |
| [nvidia/HelpSteer](https://huggingface.co/datasets/nvidia/HelpSteer) | CC BY 4.0 |
| [nvidia/HelpSteer3](https://huggingface.co/datasets/nvidia/HelpSteer3) | CC BY 4.0 |
| [oneonlee/cleansed_emocontext](https://huggingface.co/datasets/oneonlee/cleansed_emocontext) | MPL-2.0 |
| [pietrolesci/breaking_nli](https://huggingface.co/datasets/pietrolesci/breaking_nli) | CC BY-SA 4.0 |
| [pietrolesci/gen_debiased_nli](https://huggingface.co/datasets/pietrolesci/gen_debiased_nli) | CC BY-SA 4.0, OANC |
| [pietrolesci/robust_nli](https://huggingface.co/datasets/pietrolesci/robust_nli) | CC BY-SA 4.0, MIT, OANC |
| [pietrolesci/robust_nli_is_sd](https://huggingface.co/datasets/pietrolesci/robust_nli_is_sd) | OANC |
| [pietrolesci/robust_nli_li_ts](https://huggingface.co/datasets/pietrolesci/robust_nli_li_ts) | OANC |
| [pkavumba/balanced-copa](https://huggingface.co/datasets/pkavumba/balanced-copa) | CC BY 4.0, BSD-2-Clause |
| [pushpdeep/UltraFeedback-paired](https://huggingface.co/datasets/pushpdeep/UltraFeedback-paired) | MIT |
| [qbao775/PARARULE-Plus](https://huggingface.co/datasets/qbao775/PARARULE-Plus) | MIT |
| [SemEvalWorkshop/sem_eval_2010_task_8](https://huggingface.co/datasets/SemEvalWorkshop/sem_eval_2010_task_8) | CC0 1.0 |
| [sileod/discovery](https://huggingface.co/datasets/sileod/discovery) | Apache-2.0 |
| [sileod/mindgames](https://huggingface.co/datasets/sileod/mindgames) | Apache-2.0 |
| [sileod/probability_words_nli](https://huggingface.co/datasets/sileod/probability_words_nli) | Apache-2.0, CC BY-SA 4.0 |
| [stanfordnlp/snli](https://huggingface.co/datasets/stanfordnlp/snli) | CC BY-SA 4.0 |
| [tasksource/arct](https://huggingface.co/datasets/tasksource/arct) | Apache-2.0 |
| [tasksource/autotnli](https://huggingface.co/datasets/tasksource/autotnli) | Apache-2.0 |
| [tasksource/babi_nli](https://huggingface.co/datasets/tasksource/babi_nli) | BSD |
| [tasksource/clcd-english](https://huggingface.co/datasets/tasksource/clcd-english) | Apache-2.0 |
| [tasksource/conceptrules_v2](https://huggingface.co/datasets/tasksource/conceptrules_v2) | MIT, custom (see the dataset page) |
| [tasksource/corr2cause](https://huggingface.co/datasets/tasksource/corr2cause) | MIT |
| [tasksource/defeasible-nli](https://huggingface.co/datasets/tasksource/defeasible-nli) | Apache-2.0, MIT |
| [tasksource/doc-nli](https://huggingface.co/datasets/tasksource/doc-nli) | BSD-3-Clause (the package); it contains ANLI premise-hypothesis pairs, CC BY-NC 4.0 (head data only; see the note below) |
| [tasksource/dynahate](https://huggingface.co/datasets/tasksource/dynahate) | GPL |
| [tasksource/ethos](https://huggingface.co/datasets/tasksource/ethos) | AGPL-3.0 |
| [tasksource/goal-step-wikihow](https://huggingface.co/datasets/tasksource/goal-step-wikihow) | MIT (the package); its text is from wikiHow, CC BY-NC-SA 3.0 (head data only; see the note below) |
| [tasksource/google_wellformed_query](https://huggingface.co/datasets/tasksource/google_wellformed_query) | CC BY-SA 4.0 |
| [tasksource/hate_speech18](https://huggingface.co/datasets/tasksource/hate_speech18) | CC BY-SA 3.0 |
| [tasksource/help-nli](https://huggingface.co/datasets/tasksource/help-nli) | Creative Commons (version not stated), CC BY-SA 4.0 |
| [tasksource/I2D2](https://huggingface.co/datasets/tasksource/I2D2) | Apache-2.0 |
| [tasksource/implicatures](https://huggingface.co/datasets/tasksource/implicatures) | GPL |
| [tasksource/logical-entailment](https://huggingface.co/datasets/tasksource/logical-entailment) | Apache-2.0 |
| [tasksource/lonli](https://huggingface.co/datasets/tasksource/lonli) | MIT |
| [tasksource/monotonicity-entailment](https://huggingface.co/datasets/tasksource/monotonicity-entailment) | Apache-2.0 |
| [tasksource/multilingual-sentiments](https://huggingface.co/datasets/tasksource/multilingual-sentiments) | Apache-2.0 |
| [tasksource/recast](https://huggingface.co/datasets/tasksource/recast) | CC BY-SA 4.0 |
| [tasksource/resnli](https://huggingface.co/datasets/tasksource/resnli) | CC BY 4.0 |
| [tasksource/ruletaker](https://huggingface.co/datasets/tasksource/ruletaker) | Apache-2.0 |
| [tasksource/scone](https://huggingface.co/datasets/tasksource/scone) | CC0 1.0 |
| [tasksource/SDOH-NLI](https://huggingface.co/datasets/tasksource/SDOH-NLI) | CC BY 4.0 |
| [tasksource/sherliic](https://huggingface.co/datasets/tasksource/sherliic) | CC BY 4.0 |
| [tasksource/silicone](https://huggingface.co/datasets/tasksource/silicone) | CC BY-SA 4.0, CC BY 4.0 |
| [tasksource/SpaceNLI](https://huggingface.co/datasets/tasksource/SpaceNLI) | MIT |
| [tasksource/spartqa-mchoice](https://huggingface.co/datasets/tasksource/spartqa-mchoice) | MIT |
| [tasksource/spartqa-yn](https://huggingface.co/datasets/tasksource/spartqa-yn) | Apache-2.0 |
| [tasksource/starcon](https://huggingface.co/datasets/tasksource/starcon) | CC0 1.0 |
| [tasksource/stepgame](https://huggingface.co/datasets/tasksource/stepgame) | MIT |
| [tasksource/subjectivity](https://huggingface.co/datasets/tasksource/subjectivity) | MIT |
| [tasksource/syntactic-augmentation-nli](https://huggingface.co/datasets/tasksource/syntactic-augmentation-nli) | MIT |
| [tasksource/temporal-nli](https://huggingface.co/datasets/tasksource/temporal-nli) | Apache-2.0 |
| [tasksource/tomi-nli](https://huggingface.co/datasets/tasksource/tomi-nli) | GPL-3.0 |
| [tasksource/universal-joy](https://huggingface.co/datasets/tasksource/universal-joy) | GPL |
| [tweets-hate-speech-detection/tweets_hate_speech_detection](https://huggingface.co/datasets/tweets-hate-speech-detection/tweets_hate_speech_detection) | GPL-3.0 |
| [unigram/FOL-nli](https://huggingface.co/datasets/unigram/FOL-nli) | Apache-2.0 |
| [webis/args_me](https://huggingface.co/datasets/webis/args_me) | CC BY 4.0 |
| [zeroshot/twitter-financial-news-sentiment](https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment) | MIT |
| [Zhengping/UNLI](https://huggingface.co/datasets/Zhengping/UNLI) | MIT |

### Community synthetic datasets

- [tasksource/procedural-typed-decisions](https://huggingface.co/datasets/tasksource/procedural-typed-decisions)
  (tasksource, Sileo, 2024), Apache-2.0
- [frontier-infra/jebadiah-synth-v2](https://huggingface.co/datasets/frontier-infra/jebadiah-synth-v2), Apache-2.0
- [kaivoss/system-one-270m-data](https://huggingface.co/datasets/kaivoss/system-one-270m-data), Apache-2.0
- [vagmi/jevlite_dataset](https://huggingface.co/datasets/vagmi/jevlite_dataset), CC BY-SA 4.0
- [mghafiri/decision-model-scenarios](https://huggingface.co/datasets/mghafiri/decision-model-scenarios), MIT

### Non-commercial data in the head

The head's fit data (16,384 decision questions) and calibration data (4,086) include two sources whose text is non-commercial. The
model's fine-tuning data contain neither.

- **DocNLI** package (BSD-3-Clause, Yin et al., 2021), via [tasksource/doc-nli](https://huggingface.co/datasets/tasksource/doc-nli):
  70 units in the fit data and 17 in the calibration data. 10 of them (8 fit, 2 calibration) contain a complete premise-hypothesis
  pair from upstream ANLI (CC BY-NC 4.0, Nie et al., 2020), and 23 (21 fit, 2 calibration) contain ANLI text, from ANLI rounds 2
  and 3. Some of these units contain near-duplicates of 3 ANLI test items on the Decision Index suite; the items are listed under
  Overlap with evaluation data and counted as wrong in the board-rule score.
- **wikiHow**, via [tasksource/goal-step-wikihow](https://huggingface.co/datasets/tasksource/goal-step-wikihow) (MIT, the package):
  81 units in the fit data and 9 in the calibration data. Their text is from wikiHow (CC BY-NC-SA 3.0).

No dataset text is redistributed.
