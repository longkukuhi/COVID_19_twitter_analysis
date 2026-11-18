# Classifying COVID-19 tweets for crisis response

These research notebooks investigate which crisis information types a tweet conveys and how urgently an emergency manager should review it. The accompanying ISCRAM 2021 paper compares classical text classifiers with BERT on the TREC-IS 2020 COVID-19 data across 12 locations. [Publication and full text](https://eprints.gla.ac.uk/252696/)

## Explore the experiments

| Question | Notebook |
| --- | --- |
| How do traditional text classifiers perform? | [Classical machine learning experiments](scripts/COVID19_classicial_ml_31_03_2021.ipynb) |
| How does BERT classify information types? | [Final information-type experiment](scripts/COVID_Bert_information_type_final.ipynb) and [cross-validation](scripts/COVID_Bert_information_type_cross_validate.ipynb) |
| How does BERT estimate message priority? | [Final priority experiment](scripts/COVID_Bert_priority_final.ipynb) |
| How do the experiments vary across locations? | [All-locations notebook](scripts/COVID19_all_locations_final.ipynb) |

Start by viewing a notebook on GitHub to read the saved analysis. To rerun it, place the relevant tweet records and TREC-IS labels at the paths in that notebook's loading cells. The repository contains the notebooks and helper code. Original data files and cached CSVs referenced by the cells must be supplied separately.

[Dataset and evaluation helpers](scripts/covid_tools.py) cover label processing, classical classifiers, scoring and plots. [BERT helpers](scripts/BERTs.py) use PyTorch and Hugging Face Transformers. The historical notebooks import APIs such as `transformers.AdamW` and `sklearn.metrics.plot_confusion_matrix`, so check those imports when selecting an environment. [Original Colab references](scripts/colab_links.txt)

## Citation

Zijun Long and Richard McCreadie. *Automated Crisis Content Categorization for COVID-19 Tweet Streams*. Proceedings of the 18th International Conference on Information Systems for Crisis Response and Management (ISCRAM), 2021, pp. 667–678. [University of Glasgow publication record](https://eprints.gla.ac.uk/252696/)
