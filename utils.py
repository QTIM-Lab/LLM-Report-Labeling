import json
import re
from typing import Dict, List, Union
import pandas as pd
import csv


def extract_labels(
    input_string: str, CoT: bool = False
) -> List[Dict[str, Union[str, Dict[str, str]]]]:

    report_blocks = input_string.split("```")
    patients_list = []

    for block in report_blocks:
        json_str = block.strip()
        json_str = re.sub(r",\s*}", "}", json_str)
        if json_str.startswith("{") and json_str.endswith("}"):
            patient_data = json.loads(json_str)
            if CoT:
                patient_dict = {
                    condition: details["label"]
                    for condition, details in patient_data.items()
                }
                explanations_concat = " ".join(
                    [details["explanation"] for details in patient_data.values()]
                )
                patient_dict["Explanations"] = explanations_concat
            else:
                patient_dict = {
                    condition: label for condition, label in patient_data.items()
                }
                patient_dict["Explanations"] = "Not Available"

            patients_list.append(patient_dict)

    return patients_list


def extract_labels_non_gpt(
    input_string: str, CoT: bool = False
) -> List[Dict[str, Union[str, Dict[str, str]]]]:

    patients_list = []

    matches = re.findall(r"\{[\s\S]*?\}", input_string)

    for json_str in matches:
        try:
            json_str = json_str.strip()
            json_str = re.sub(r",\s*}", "}", json_str)
            patient_data = json.loads(json_str)
            if not patient_data:
                print(f"Empty JSON object found: {json_str}")
                continue
            if CoT:
                patient_dict = {
                    condition: details["label"]
                    for condition, details in patient_data.items()
                }
                explanations_concat = " ".join(
                    [details["explanation"] for details in patient_data.values()]
                )
                patient_dict["Explanations"] = explanations_concat
            else:
                patient_dict = {
                    condition: label for condition, label in patient_data.items()
                }
                patient_dict["Explanations"] = "Not Available"

            patients_list.append(patient_dict)
            break

        except json.JSONDecodeError:
            pass

    if not patients_list:
        print(f"Empty patients list from this json str: {input_string}")
    return patients_list


def write_result_to_file(result, output_file):
    with open(output_file, 'a', newline='') as file:
        fieldnames = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices', 'Explanation', 'text', 'Accession Number', 'Patient MRN']
        writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        
        for res in result:
            # Ensure all keys are present and escape newline characters
            processed_res = {key: (str(res.get(key, "")).replace("\n", "\\n") if res.get(key) is not None else "") for key in fieldnames}
            writer.writerow(processed_res)


def load_and_preprocess_data(file_path, few_shot_list=None):
    df = pd.read_csv(file_path)
    if few_shot_list:
        df = df[~df['acc'].isin(few_shot_list)]
    df = df.sort_values(by='acc').reset_index(drop=True)
    df.rename(columns={'text': 'Report Text'}, inplace=True)
    df.columns = df.columns.str.replace(' ', '_')
    df = df.astype(str)
    df.fillna('No', inplace=True)

    return df

def update_column(df, column, condition_cols):
    mask = (df[column] == 'No') & (df[condition_cols].apply(lambda x: 'Yes' in x.values, axis=1))
    df.loc[mask, column] = 'Yes'

def calculate_performance(y_true, y_pred, metric_func, average=None, pos_label="Yes"):
    performance_scores = {}
    for col in y_true.columns:
        if average is not None:
            performance_scores[col] = metric_func(y_true[col], y_pred[col], average='micro', zero_division=0)
        else:
            performance_scores[col] = metric_func(y_true[col], y_pred[col])
    return performance_scores