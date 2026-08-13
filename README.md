# sentence_to_ttp_augmentation

This repository contains the code and intermediate artifacts for the paper [`sentence_to_ttp.pdf`](./sentence_to_ttp.pdf), which studies how to improve sentence-to-TTP mapping by making better use of fragmented supervision sources in cyber threat intelligence (CTI).

The paper focuses on sentence-level mapping from CTI text to MITRE ATT&CK techniques and shows that performance can be improved without collecting new manual annotations. The main idea is to recover and transfer supervision that already exists in other forms, such as ATT&CK text and document-level CTI labels.

## What The Paper Contributes

The paper introduces two complementary augmentation strategies:

- `Label-space recovery`: reuse ATT&CK-derived supervision that would otherwise be discarded because benchmark label spaces are narrower than the full ATT&CK ontology.
- `TTP-Transfer`: a Siamese bi-encoder retrieval method that grounds document-level CTI labels to candidate evidence sentences, turning weak report-level supervision into sentence-level training pairs.

Across the two benchmark datasets used in the paper, `TRAM` and `AnnoCTR`, the combined augmentation strategy reports improvements of `3.9%` to `12.1%`.

## Repository Overview

This repository is organized around the paper pipeline:

- [`classification/`](./classification/) contains the sentence-to-TTP classifiers, training scripts, test scripts, benchmark datasets, and analysis notebooks.
- [`data_augmentatio/`](./data_augmentatio/) contains the data construction and augmentation workflow, including sentence extraction, audit samples, CTI report processing, and Siamese retrieval notebooks and scripts.
- [`llm_generation/`](./llm_generation/) is reserved for LLM-based generation utilities and supporting assets.
- [`sentence_to_ttp.pdf`](./sentence_to_ttp.pdf) is the paper describing the methodology and reported results.

## How This Maps To The Paper

At a high level, the repository follows the same stages described in the paper:

1. `Dataset characterization and collection`
   The paper analyzes supervision fragmentation across sentence-level, document-level, and ATT&CK-derived resources.

2. `Supervision recovery from ATT&CK`
   The repository contains ATT&CK-derived augmentation assets and processed datasets used to recover supervision through hierarchy-aware and semantic matching.

3. `Document-to-sentence transfer`
   The Siamese retrieval workflow in [`data_augmentatio/`](./data_augmentatio/) is used to ground document-level CTI labels to candidate evidence sentences.

4. `Sentence-level classification`
   The supervised and similarity-based classifiers in [`classification/`](./classification/) are used to evaluate the resulting augmented datasets on `TRAM` and `AnnoCTR`.

## Main Artifacts In This Repository

Examples of relevant checked-in artifacts include:

- benchmark and augmented datasets in [`classification/datasets/`](./classification/datasets/)
- classifier training and evaluation code such as [`classification/train_multilabel.py`](./classification/train_multilabel.py), [`classification/test_labeled.py`](./classification/test_labeled.py), and [`classification/unlabeled.py`](./classification/unlabeled.py)
- augmentation and retrieval scripts such as [`data_augmentatio/siamese_test_only_all_labels.py`](./data_augmentatio/siamese_test_only_all_labels.py)
- notebooks used during data preparation, augmentation, evaluation, and analysis

## Reproducing The Paper

This is currently a research repository rather than a polished benchmark package. The code and many intermediate datasets are present, but some large artifacts are intentionally not tracked in git because they exceed GitHub file size limits, including:

- local Python environments
- fine-tuned model checkpoints
- large cached runtime libraries
- some generated experiment outputs

If you want to reproduce the experiments, the best starting point is:

1. Read the workflow notes in [`classification/README.md`](./classification/README.md).
2. Inspect the datasets available in [`classification/datasets/`](./classification/datasets/).
3. Review the augmentation assets and retrieval notebooks in [`data_augmentation/`](./data_augmentatio/).
4. Recreate the environment from [`classification/requirements.txt`](./classification/requirements.txt) in a fresh virtual environment rather than using the bundled local environment folders.

## Notes On The Current State Of The Repo

- The repository reflects active research code and includes exploratory notebooks, generated outputs, and dataset snapshots.
- Some directory names and scripts still reflect development-time naming choices, for example `data_augmentation/`.
- The current root README is intended to explain the paper-to-code mapping clearly; the lower-level execution details are still mostly documented inside [`classification/README.md`](./classification/README.md).

