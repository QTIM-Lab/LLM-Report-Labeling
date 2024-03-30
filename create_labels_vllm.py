import argparse
import json
import math
import os
import time
import sys

import pandas as pd
import prompts
import utils
from openai import OpenAI
from pycrumbs import tracked
from tqdm import tqdm
import concurrent.futures
import csv


def process_batch(args):
    batch_data, prompt_type, max_retries, verbose, model, num_few_shots = args
    reports = batch_data["Report Text"].values.tolist()
    accs = batch_data["Accession Number"].tolist()
    mrns = batch_data["Patient MRN"].tolist()

    if prompt_type.strip().lower() == "zero-shot":
        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = prompts.system_prompt_basic + prompt
        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "zero-shot-imagenome":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = prompts.system_prompt_basic_imagenome + prompt
        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "few-shot":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = (
            prompts.system_prompt_basic + prompts.few_shot_examples_test_set + prompt
        )

        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "few-shot-positive":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = (
            prompts.system_prompt_basic
            + prompts.generate_positive_few_shot_examples_mgb(num_few_shots)[0]
            + prompt
        )

        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "few-shot-random":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = (
            prompts.system_prompt_basic
            + prompts.generate_few_shot_examples_mgb(num_few_shots)[0]
            + prompt
        )

        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "few-shot-imagenome-positive":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = (
            prompts.system_prompt_basic_imagenome
            + prompts.generate_few_shot_examples_imagenome_positive(num_few_shots)[0]
            + prompt
        )

        messages = [
            {"role": "user", "content": prompt_total},
        ]

    elif prompt_type.lower() == "few-shot-imagenome-random":

        prompt = prompts.generate_user_prompt(reports, CoT=False)

        prompt_total = (
            prompts.system_prompt_basic_imagenome
            + prompts.generate_few_shot_examples_imagenome(num_few_shots)[0]
            + prompt
        )

        messages = [
            {"role": "user", "content": prompt_total},
        ]

    else:

        raise ValueError("Invalid prompt type specified.")

    if verbose:
        print("Verbose mode is on")
        for m in messages:
            print(f"\n{m['role']} prompt:")
            print(m["content"])

    successful = False  # Flag to check if the batch was processed successfully
    for retry in range(max_retries):
        try:

            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0,
                frequency_penalty=0,
                presence_penalty=0,
                top_p=0.95,
                # max_tokens=2000,
            )
            output_string = response.choices[0].message.content
            if "}" not in output_string:
                raise Exception(f"Incorrect Output Format:\n {output_string}")

            cleaned_strings = utils.extract_labels_non_gpt(output_string)
            for i, report in enumerate(cleaned_strings):
                report['text'] = reports[i] 
                report['Accession Number'] = accs[i]      
                report['Patient MRN'] = mrns[i]     

            if verbose:
                for report in cleaned_strings:
                    print(report)

            successful = True
            return cleaned_strings

        except Exception as e:
            print(
                f"An error occurred while processing: {messages}. Retrying. Error: {e}"
            )
            time.sleep(5)
            continue

    if not successful:
        print(60 * "-")
        print(
            f"Failed to process batch {messages} after {max_retries} retries. Moving on to the next batch."
        )
        cleaned_strings = [
                {
                    "Atelectasis": "Failed",
                    "Pleural Effusion": "Failed",
                    "Pneumonia": "Failed",
                    "Pneumothorax": "Failed",
                }
            ]
        
        for i, report in enumerate(cleaned_strings):
                report['text'] = reports[i] 
                report['Accession Number'] = accs[i]      
                report['Patient MRN'] = mrns[i]     
   
        return cleaned_strings



@tracked(directory_parameter="output_dir", seed_parameter="seed")
def main(
    seed: int,
    num_reports: int,
    batch_size: int,
    output_dir: str,
    input_file: str,
    prompt_type: str,
    verbose: bool,
    use_multiprocessing: bool,
    model: str,
    num_few_shots: int,
) -> None:

    """
    Main function to process x-ray reports and generate labels based on the chexpert convention using GPT.

    This function reads whole reports from a CSV file, processes them in batches using
    open-source LLMs, and generates the 14 chexpert labels for them which are then saved to a CSV file in
    the specified output directory.

    Args:
        seed (int): Seed for to ensure reproducibility for random sampling.
        num_reports (int): The total number of reports to process.
        batch_size (int): The number of reports to process in each batch.
        output_dir (str): The directory where the output CSV file will be saved.
        input_file (str): The file path of the input CSV file containing the reports.
        prompt_type (str): The type of prompt to use for the GPT model. Options include 'zero-shot',
            'zero-shot-imagenome',  'few-shot', 'few-shot-positive', 'few-shot-random',
            'few-shot-imagenome-positive', and 'few-shot-imagenome-random'.
        num_few_shots (int, optional): Number of few-shot examples to use.
            For 'few-shot-random', this is the total number of few-shots.
            For prompts with positive examples, this is the number of positive examples per finding. Defaults to 10.
        verbose (bool): A flag to toggle verbose mode for detailed logging.
        model (str): LLM that will be used to run Inference on.
        num_few_shots (int): Number of few-shot examples to use for the 'few-shot' prompt.
        use_multiprocessing (bool): A flag to toggle multiprocessing for parallel processing of batches.
            By default, this uses 32 processes.

    Returns:
        None
    """

    # Key is empty for local host, but still required as a parameter
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"

    global client
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
    )

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    full_reports = pd.read_csv(input_file)

    if seed is not None:
        full_reports = full_reports.sample(frac=1, random_state=seed).reset_index(
            drop=True
        )
    else:
        full_reports = full_reports.sample(frac=1).reset_index(drop=True)

    if len(full_reports) < num_reports:
        print(
            f"""Number of reports requests exceeds the number of reports in the input csv,
        using all available reports ({len(full_reports)}) instead"""
        )
        num_reports = len(full_reports)

    max_retries = 10

    num_batches = math.ceil(num_reports / batch_size)
    print(f"number of batches: {num_batches}")

    print(f"Using the {prompt_type} prompt!")

    # save few-shot examples to a file
    prompt_type_lower = prompt_type.lower()
    if prompt_type_lower == "few-shot-imagenome":
        generator_function = prompts.generate_few_shot_examples_imagenome
    elif prompt_type_lower == "few-shot-imagenome-positive":
        generator_function = prompts.generate_few_shot_examples_imagenome_positive
    elif prompt_type_lower == "few-shot-random":
        generator_function = prompts.generate_few_shot_examples_mgb
    elif prompt_type_lower == "few-shot-positive":
        generator_function = prompts.generate_positive_few_shot_examples_mgb
    else:
        generator_function = None

    # Generate few-shot examples and save to a file if a valid generator is identified
    if generator_function:
        few_shot_examples_accs = generator_function(num_few_shots)[1]
        with open(os.path.join(output_dir, "few_shot_examples_acc.json"), "w") as f:
            json.dump(few_shot_examples_accs, f)

    batches = [
        full_reports.iloc[
            i * batch_size : min((i + 1) * batch_size, num_reports)  # noqa: E203
        ]
        for i in range(num_batches)
    ]

    args_list = [
        (
            batch,
            prompt_type,
            max_retries,
            verbose,
            model,
            num_few_shots,
        )
        for batch in batches
    ]

    output_file = os.path.join(output_dir, "llm_labeled_reports.csv")

    with open(output_file, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema', 'Enlarged Cardiomediastinum', 'Fracture', 'Lung Lesion', 'Lung Opacity', 'Pleural Effusion', 'Pleural Other', 'Pneumonia', 'Pneumothorax', 'Support Devices', 'Explanation', 'text', 'Accession Number', 'Patient MRN']) 
        writer.writeheader()


    executor_class = concurrent.futures.ThreadPoolExecutor
    max_workers = 32 if use_multiprocessing else 1
   
    with executor_class(max_workers=max_workers) as executor:
        future_to_batch = {executor.submit(process_batch, args): args for args in args_list}

        try: 
            for future in tqdm(concurrent.futures.as_completed(future_to_batch), total=len(future_to_batch)):
                result = future.result()
                utils.write_result_to_file(result, output_file)

        except KeyboardInterrupt:
            print("\nInterrupt received, cancelling tasks...")
            for future in future_to_batch:
                future.cancel()
            executor.shutdown(wait=False)
            print("Cancelled pending tasks and shutting down executor.")
            sys.exit(1)  # Exit the script or handle the interrupt as needed

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for deterministic sampling"
    )
    parser.add_argument(
        "--num_reports",
        type=int,
        required=True,
        help="Total number of reports to be processed",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Number of reports to include in each request",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to store the results and response",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Directory to read the reports from",
    )
    parser.add_argument(
        "--prompt_type",
        type=str,
        required=True,
        choices=[
            "zero-shot",
            "zero-shot-imagenome",
            "few-shot",
            "few-shot-positive",
            "few-shot-random",
            "few-shot-imagenome-positive",
            "few-shot-imagenome-random",
        ],
        help="Style of prompt to use. Choose from 'zero-shot', 'zero-shot-imagenome', "
        "'few-shot', 'few-shot-positive', 'few-shot-random', 'few-shot-imagenome-positive', "
        "'few-shot-imagenome-random'",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The language model to use"
    )
    parser.add_argument(
        "--num_few_shots",
        type=int,
        default=10,
        help="Number of few-shot examples to use, for the random few-shots this is the total "
        "number of few-shots, for the few-shots with positive examples this is the number "
        "of positive examples per finding. Default is 10.",
    )
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")
    parser.add_argument(
        "--use_multiprocessing",
        action="store_true",
        help="Run code with multiprocessing, by default this spawns 32 processes",
    )

    args = parser.parse_args()

    args = parser.parse_args()
    main(**vars(args))