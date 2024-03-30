import argparse
import json
import math
import os
import time

import openai
import pandas as pd
import prompts
import utils
from pycrumbs import tracked
from tqdm import tqdm


@tracked(directory_parameter="output_dir", seed_parameter="seed")
def main(
    seed: int,
    num_reports: int,
    batch_size: int,
    output_dir: str,
    input_file: str,
    prompt_type: str,
    num_few_shots: int,
    model: str,
    verbose: bool,
) -> None:

    """
    Main function to process x-ray reports and generate labels based on the chexpert convention using GPT.

    This function reads whole reports from a CSV file, processes them in batches using
    GPT-4, and generates the 14 chexpert labels for them which are then saved to a CSV file in
    the specified output directory.

    Args:
        seed (int): Seed for to ensure reproducibility for random sampling.
        num_reports (int): The total number of reports to process.
        batch_size (int): The number of reports to process in each batch. Defaults to 1. 
        output_dir (str): The directory where the output CSV file will be saved.
        input_file (str): The file path of the input CSV file containing the reports.
        prompt_type (str): The type of prompt to use for the GPT model. Options include 'zero-shot',
            'zero-shot-imagenome',  'few-shot', 'few-shot-positive', 'few-shot-random',
            'few-shot-imagenome-positive', and 'few-shot-imagenome-random'.
        num_few_shots (int, optional): Number of few-shot examples to use.
            For 'few-shot-random', this is the total number of few-shots.
            For prompts with positive examples, this is the number of positive examples per finding. Defaults to 10.
        model (str): LLM that will be used to run Inference on.
        verbose (bool): A flag to toggle verbose mode for detailed logging.

    Returns:
        None
    """

    openai.api_type = "azure"

    if model.lower() == "gpt-4":
        openai.api_base = "ENTER YOU API BASE URL HERE"
        openai.base_url = "ENTER YOU API BASE URL HERE"
        openai.api_version = "2023-07-01-preview"
        engine = "gpt-4"

    elif model.lower() == "gpt-3.5":
        openai.api_base = "ENTER YOU API BASE URL HERE"
        openai.api_version = "2023-03-15-preview"
        engine = "TDGPT35TURBO16k"

    else:
        raise ValueError(f"Invalid model specified. Model <{model}> is not supported.")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    if "CHATGPT_API_KEY" not in os.environ:
        raise KeyError("Need to set CHATGPT_API_KEY environment variable.")
    else:
        openai.api_key = os.environ["CHATGPT_API_KEY"]

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

    num_batches = math.ceil(num_reports / batch_size)
    max_retries = 3

    num_batches = math.ceil(num_reports / batch_size)
    print(f"number of batches: {num_batches}")
    all_dfs = []

    CoT = "cot" in prompt_type.lower()

    for i in tqdm(range(num_batches)):

        start_idx = i * batch_size
        end_idx = min((i + 1) * batch_size, num_reports)

        batch_data = full_reports.iloc[start_idx:end_idx]
        reports = batch_data["Report Text"].values.tolist()
        accs = batch_data["acc"].tolist()
        mrns = batch_data["mrn"].tolist()

        if prompt_type.lower() == "zero-shot":
            if i == 0:
                print("Using the zero-shot prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)
            messages = [
                {"role": "system", "content": prompts.system_prompt_basic},
                {"role": "user", "content": prompt},
            ]
        elif prompt_type.lower() == "zero-shot-imagenome":
            if i == 0:
                print("Using the zero-shot-imagenome prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)
            messages = [
                {"role": "system", "content": prompts.system_prompt_basic_imagenome},
                {"role": "user", "content": prompt},
            ]

        elif prompt_type.lower() == "few-shot":
            if i == 0:
                print("Using the few-shot prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)

            system_prompt = (
                prompts.system_prompt_basic + prompts.few_shot_examples_test_set
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        elif prompt_type.lower() == "few-shot-positive":
            if i == 0:
                print("Using the few-shot-positive prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)

            system_prompt = (
                prompts.system_prompt_basic
                + prompts.generate_positive_few_shot_examples_mgb(num_few_shots)[0]
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        elif prompt_type.lower() == "few-shot-random":
            if i == 0:
                print("Using the few-shot-random prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)

            system_prompt = (
                prompts.system_prompt_basic
                + prompts.generate_few_shot_examples_mgb(num_few_shots)[0]
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        elif prompt_type.lower() == "few-shot-imagenome-positive":
            if i == 0:
                print("Using the few-shot-imagenome-positive prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)

            # fmt: off
            system_prompt = (
                prompts.system_prompt_basic_imagenome
                + prompts.generate_few_shot_examples_imagenome_positive(num_few_shots)[0]
            )
            # fmt: on

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
        elif prompt_type.lower() == "few-shot-imagenome-random":
            if i == 0:
                print("Using the few-shot-imagenome-random prompt")
            prompt = prompts.generate_user_prompt(reports, CoT=False)

            system_prompt = (
                prompts.system_prompt_basic_imagenome
                + prompts.generate_few_shot_examples_imagenome(num_few_shots)[0]
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]

        else:
            raise ValueError(
                f"Invalid prompt type specified. Type <{prompt_type}> not supported."
            )

        if verbose:
            print("Verbose mode is on")
            for m in messages:
                print(f"\n{m['role']} prompt:")
                print(m["content"])

        successful = False  # Flag to check if the batch was processed successfully
        for retry in range(max_retries):
            try:
                response = openai.ChatCompletion.create(
                    engine=engine,
                    messages=messages,
                    temperature=0.0,
                    # max_tokens=5000,
                    top_p=0.95,
                    frequency_penalty=0,
                    presence_penalty=0,
                    stop=None,
                )

                output_string = response["choices"][0]["message"]["content"]
                cleaned_strings = utils.extract_labels(output_string, CoT=CoT)

                if verbose:
                    print(f"these are the cleaned strings:{cleaned_strings}")

                df = pd.DataFrame(cleaned_strings)
                df["text"] = reports
                col_to_move = df.pop("text")
                df.insert(0, "text", col_to_move)
                df.insert(1, "acc", accs)
                df.insert(2, "mrn", mrns)
                all_dfs.append(df)
                if verbose:
                    print(df)

                successful = True
                break

            except Exception as e:
                print(
                    f"An error occurred while processing batch {i}. Retrying. Error: {e}"
                )
                time.sleep(5)
                continue

        if not successful:
            print(60 * "-")
            print(
                f"Failed to process batch {i} after {max_retries} retries. Moving on to the next batch."
            )

    final_df = pd.concat(all_dfs, ignore_index=True)
    final_df.to_csv(os.path.join(output_dir, "gpt_labeled_reports.csv"), index=False)

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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process some arguments.")
    parser.add_argument(
        "--seed", type=int, default=None, help="A seed for deterministic sampling"
    )
    parser.add_argument(
        "--num_reports",
        type=int,
        required=True,
        help="How many reports in should be processed in total",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="How many reports should be included in one request",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for results and response",
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input directory to read reports from",
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
        "--num_few_shots",
        type=int,
        default=10,
        help="Number of few-shot examples to use, for the random few-shots this is the total "
        "number of few-shots, for the few-shots with positive examples this is the number "
        "of positive examples per finding. Default is 10.",
    )
    parser.add_argument(
        "--model", type=str, required=True, help="The language model to use"
    )
    parser.add_argument("--verbose", action="store_true", help="Run in verbose mode")

    args = parser.parse_args()
    main(**vars(args))
