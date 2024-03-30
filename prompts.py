from typing import List
import pandas as pd 
import numpy as np
import json


def generate_user_prompt(reports_list: List, CoT: bool = False) -> str:

    if CoT:
        output = (
            '"""\nClassify the following radiology reports according to the template, '
        )
        output += (
            "provide a new template for each report along with your explanation:\n"
        )
    else:
        output = (
            '"""\nClassify the following radiology report according to the template.'
        )
        output += "Always output the full template, even if a finding is not mentioned:\n"


    for i, row in enumerate(reports_list):
        output += "```\n"
        output += f"{row}\n"
        output += "```\n"
        output += 30 * "---"

    output += '\n"""'

    return output


system_prompt_basic = """
Please accurately classify radiology reports for the presence
or absence of findings. For each report, you will classify for the presence or absence of the following
findings: Cardiac congestion, lung opacities (that includes pneumonia, atelectasis, edema, consolidation and
lung lesions), pleural effusion (this does NOT include pericardial effusion), other pleural abnormalities,
pneumothorax, presence of support devices (this includes: thoracic drains, venous catheters, gastric
tubes, tracheal tubes, artificial valves and other implanted devices or tubes, it does NOT include nipple markers,
ecg leads, suture lines or fixation hardware). structure your answer like the
template I provide to you delimited by triple backticks and return this template and nothing else. 
ALWAYS RETURN THE FULL TEMPLATE:

```
{"Atelectasis": "[ANSWER]", "Cardiomegaly": "[ANSWER]", "Consolidation": "[ANSWER]", "Edema": "[ANSWER]", "Enlarged Cardiomediastinum": "[ANSWER]", "Fracture": "[ANSWER]", "Lung Lesion": "[ANSWER]", "Lung Opacity": "[ANSWER]", "Pleural Effusion": "[ANSWER]", "Pleural Other": "[ANSWER]", "Pneumonia": "[ANSWER]", "Pneumothorax": "[ANSWER]", "Support Devices": "[ANSWER]"}
```

the default answer is 'Undefined'.
if the patient has the finding, answer 'Yes'.
if the patient does not have the finding, answer 'No'.
if finding unclear, answer 'Maybe'.

"""

system_prompt_basic_imagenome = """
Please accurately classify radiology reports for the presence
or absence of findings. For each report, you will classify for the presence or absence of the following
findings: atelectasis, pleural effusion (this does NOT include pericardial effusion), 
pneumothorax and pneumonia. structure your answer like the
template I provide to you delimited by triple backticks and return this template and nothing else. 
ALWAYS RETURN THE FULL TEMPLATE:

```
{"Atelectasis": "[ANSWER]", "Pleural Effusion": "[ANSWER]", "Pneumonia": "[ANSWER]", "Pneumothorax": "[ANSWER]"}
```

the default answer is 'Undefined'.
if the patient has the finding, answer 'Yes'.
if the patient does not have the finding, answer 'No'.

"""

diseases_keys_imagenome = {
    "Atelectasis": "Atelectasis",
    "Pleural_Effusion": "Pleural Effusion",  
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax"
}

def format_row_imagenome(row):
    disease_mapping = {
        'Yes': "Yes",
        'No': "No",
        np.nan: "No",
        'maybe': 'Maybe',
        1: "Yes",
        0: "No"
    }

    diseases = {diseases_keys_imagenome[disease]: disease_mapping.get(row[disease], "No") for disease in diseases_keys_imagenome}
    diseases_json = json.dumps(diseases)

    report_text = row['Report Text'] if not pd.isna(row['Report Text']) else "Not Available"
    report_acc = row['acc'] if not pd.isna(row['acc']) else "Not Available"

    formatted_string = f"Report:\n{report_text}\n\n\n\nAnswer:\n```{diseases_json}``` \n\n\n\n-----------------------------"
    return formatted_string, report_acc

def generate_few_shot_examples_imagenome(num_samples=10):
    imagenome_df = pd.read_csv("/path/to/few_shot_reports_imagenome.csv")
    few_shot_df = imagenome_df.sample(num_samples, random_state=123)
    formatted_results = few_shot_df.apply(format_row_imagenome, axis=1)
    
    formatted_strings = [result[0] for result in formatted_results]
    report_accs = [result[1] for result in formatted_results]
    
    few_shot_examples_imagenome = '\n'.join(formatted_strings)
    return few_shot_examples_imagenome, report_accs

def generate_few_shot_examples_imagenome_positive(num_samples=1, random_state=123):
    imagenome_df = pd.read_csv("/path/to/few_shot_reports_imagenome.csv")
    # Convert 'Yes', 'No', 'maybe', and np.nan to numerical values for easier processing
    for disease in diseases_keys_imagenome:
        imagenome_df[disease] = imagenome_df[disease].map({'yes': 1, 'no': 0, np.nan: 0, 'maybe': 0})
    imagenome_df['Positive_Labels'] = imagenome_df[list(diseases_keys_imagenome.keys())].sum(axis=1)
    sampled_rows = []
    covered_diseases = set()
    # Sort dataframe by the number of positive labels in descending order
    imagenome_df_sorted = imagenome_df.sort_values(by='Positive_Labels', ascending=False)

    for _, row in imagenome_df_sorted.iterrows():
        if len(row['Report Text']) > 1000:
            continue  
        # Correctly access each disease's status in the row using diseases_keys keys
        current_diseases = {disease for disease in diseases_keys_imagenome if row[disease] == 1}
        # Check if the report adds new diseases to the coverage
        if not current_diseases.issubset(covered_diseases):
            sampled_rows.append(row)
            covered_diseases.update(current_diseases)
        # Break if all diseases are covered or if we've reached the desired number of samples
        if len(covered_diseases) == len(diseases_keys_imagenome) or len(sampled_rows) >= num_samples * len(diseases_keys_imagenome):
            break

    few_shot_df = pd.DataFrame(sampled_rows).drop_duplicates()
    formatted_results = few_shot_df.apply(format_row_imagenome, axis=1)

    formatted_strings = [result[0] for result in formatted_results]
    report_accs = [result[1] for result in formatted_results]

    few_shot_examples_imagenome = '\n'.join(formatted_strings)
    return few_shot_examples_imagenome, report_accs

diseases_keys = {
    "Atelectasis": "Atelectasis",
    "Cardiomegaly": "Cardiomegaly",
    "Consolidation": "Consolidation",
    "Edema": "Edema",
    "Enlarged_Cardiomediastinum": "Enlarged Cardiomediastinum",
    "Fracture": "Fracture",
    "Lung_Lesion": "Lung Lesion",
    "Lung_Opacity": "Lung Opacity",
    "Pleural_Effusion": "Pleural Effusion",
    "Pleural_Other": "Pleural Other",
    "Pneumonia": "Pneumonia",
    "Pneumothorax": "Pneumothorax",
    "Support_Devices": "Support Devices"
}


def generate_few_shot_examples_mgb(num_samples=10):
    imagenome_df = pd.read_csv("/path/to/few_shot_reports_mgb.csv")
    few_shot_df = imagenome_df.sample(num_samples, random_state=123)
    formatted_results = few_shot_df.apply(format_row_mgb, axis=1)
    
    formatted_strings = [result[0] for result in formatted_results]
    report_accs = [result[1] for result in formatted_results]
    
    few_shot_examples_imagenome = '\n'.join(formatted_strings)
    return few_shot_examples_imagenome, report_accs

def format_row_mgb(row):
    disease_mapping = {
        'Yes': "Yes",
        'No': "No",
        np.nan: "No",
        'maybe': 'Maybe',
        1: "Yes",
        0: "No"
    }

    diseases = {diseases_keys[disease]: disease_mapping.get(row[disease], "No") for disease in diseases_keys}
    diseases_json = json.dumps(diseases)

    report_text = row['Report Text'] if not pd.isna(row['Report Text']) else "Not Available"
    report_acc = row['acc'] if not pd.isna(row['acc']) else "Not Available"

    formatted_string = f"Report:\n{report_text}\n\n-----------------------------\n\nAnswer:\n```{diseases_json}``` \n\n"
    return formatted_string, report_acc


def generate_positive_few_shot_examples_mgb(num_samples=1):
    imagenome_df = pd.read_csv("/path/to/few_shot_reports_mgb.csv")
    # Convert 'Yes', 'No', 'maybe', and np.nan to numerical values for easier processing
    for disease in diseases_keys:
        imagenome_df[disease] = imagenome_df[disease].map({'Yes': 1, 'No': 0, np.nan: 0, 'maybe': 0})
    imagenome_df['Positive_Labels'] = imagenome_df[list(diseases_keys.keys())].sum(axis=1)

    sampled_rows = []
    covered_diseases = set()
    # Sort dataframe by the number of positive labels in descending order
    imagenome_df_sorted = imagenome_df.sort_values(by='Positive_Labels', ascending=False)

    for _, row in imagenome_df_sorted.iterrows():
        if len(row['Report Text']) > 1000:
            continue  
        # Correctly access each disease's status in the row using diseases_keys keys
        current_diseases = {disease for disease in diseases_keys if row[disease] == 1}
        # Check if the report adds new diseases to the coverage
        if not current_diseases.issubset(covered_diseases):
            sampled_rows.append(row)
            covered_diseases.update(current_diseases)
        # Break if all diseases are covered or if we've reached the desired number of samples
        if len(covered_diseases) == len(diseases_keys) or len(sampled_rows) >= num_samples * len(diseases_keys):
            break

    few_shot_df = pd.DataFrame(sampled_rows).drop_duplicates()
    formatted_results = few_shot_df.apply(format_row_mgb, axis=1)

    formatted_strings = [result[0] for result in formatted_results]
    report_accs = [result[1] for result in formatted_results]

    few_shot_examples_imagenome = '\n'.join(formatted_strings)
    return few_shot_examples_imagenome, report_accs