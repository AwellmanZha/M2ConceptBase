# M<sup>2</sup>ConceptBase
This repository contains the data and code for our CIKM 2024 paper "M2ConceptBase: A Fine-grained Aligned Concept-Centric Multimodal Knowledge Base".

# Data download and organization
The relevant data for knowledge base construction and downstream applications can be downloaded from [here](https://drive.google.com/drive/folders/1g7XSqB5-uJV8kTUQ7GAZDExqWG3PYjUK), and the data is organized as follows:

- candidate_concept_mining: `word_freq_train_data_230531.json`, `word_pos_freq_train_data_230514.txt`, `word_freq_train_data_230514_v7.json`.
- description_completion: `concept_description_dict_filtered_stage_1.json`, `concept_description_dict_filtered_stage_2_v1.json`, `concept_description_dict_filtered.json`, `concept_description_score_dict.json`, `m2conceptbase_concepts.json`, `grounded_concepts.json`.
- data_sources: `abstract_concepts_descriptions_ensogou.json`, `concrete_concepts_descriptions_ensogou.json`.
- okvqa: `tag2description.json`, `test_vqa_result_blip2.json`, `test_vqa_result_blip2_kg_llm.json`, `test_vqa_result_pnp_vqa.json`, `test_vqa_result_pnp_vqa_kg_llm.json`, `vqa_val_eval.json`
  - PICa: `PICa.zip`
- mm_rag: `concept_desc_dict.json`
  - mm_concept_base: `mm_concept_base.zip`
  - test: `test.zip`
  - model_answers: `model_answers.zip`


The data for **M<sup>2</sup>ConceptBase** is contained in `m2conceptbase_data.zip`, with the images encoded in Base64 format.
