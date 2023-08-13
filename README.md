# MLCopenhagen
ML model that predicts whether a patient has CRC or not based on their gut microbiome composition
The model is also tested for robustness.
The model also identifies the biomarker species of a cohort, that distinguish a patient with CRC from a control individual.

Models were built using 5 different learning algorithms.
Each model was trained on one cohort and then tested on the rest. This tests the robustness of the model.
Each model has a feature importance plot, to depict the biomarker species for a cohort.

Each .py file has the code for each model trained on one cohort of patients and then tested for robustness.

