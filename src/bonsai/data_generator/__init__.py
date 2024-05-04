from requests import post
from os import environ
from time import sleep
from json import loads, dumps
from datetime import datetime


def _get_delay():
    """
    Generates a 2-increasing integer
    """
    counter = 0

    while True:
        yield counter
        counter += 2


NORMATIVE_INSTRUCTIONS = (
    'A normative sentence is a sentence that is '
    'semantically equivalent to a sentence that has '
    '"must", "should" or "may" but does not have '
    '"must", "should" or "may". Please  generate '
    '{sentences} normative sentences that are all '
    'different, semantically equivalent to a '
    'sentence that has "{verb}", and are all '
    'related to specifications of APIs and SDKs. '
    'Please also generate {sentences} sentences '
    'that are semantically equivalent to the '
    'previous {sentences} sentences but using '
    '"{verb}" in these  sentences. Please write all '
    'these sentences without numbering or adding '
    'anything else to them. Do not add a period at the'
    ' end of each sentence.'
)

NON_NORMATIVE_INSTRUCTIONS = (
    'A non-normative sentence is a sentence that is '
    'not semantically equivalent to a sentence that has '
    '"must", "should" or "may". Please'
    ' generate {sentences} non-normative sentences that are '
    'all different and are all related to specifications of '
    'APIs and SDKs. Please write all these sentences '
    'without numbering or adding anything else to them.'
)


def generate_sentences(
    verb,
    sentences,
    run_start,
    run_end,
    results_path
):

    """
    This function generates normative or non-normative sentences using the
    ChatGPT API.

    In order to use this function, you need an API key, which is located at
    .openai/keys. This key needs to be exported to the OPENAI_API_KEY
    environment variable before executing this function.

    To generate a new API key, log into your accout at platform.openai.com.

    You will find your keys at the API keys tab at the left or at
    platform.openai.com/api-keys.

    Before using this function you will need funds in your account, check your
    funds at platform.openai.com/usage.
    """

    if verb == "NON":
        instructions = NON_NORMATIVE_INSTRUCTIONS

    else:
        instructions = NORMATIVE_INSTRUCTIONS

    api_key = environ.get("OPENAI_API_KEY")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    endpoint_url = (
        "https://api.openai.com/v1/chat/completions"
    )

    for run in range(run_start, run_end):

        json = {
            "model": "gpt-4",
            "max_tokens": 8192 - 142,
            "messages": [
                {
                    "role": "user",
                    "content": instructions.format(**locals())
                }
            ]
        }

        print()
        print(f"Generating {verb} {run}")

        for delay in _get_delay():
            try:
                start = datetime.now()
                response = post(
                    endpoint_url,
                    headers=headers,
                    json=json,
                    timeout=60
                )
                end = datetime.now()
                print(f"Took {(end - start).seconds}s")
                if response.status_code != 200:
                    raise Exception(response.content)
                break
            except Exception as error:
                print(error)
                print(f"Waiting for {delay}s")
                sleep(delay)

        content = response.content.decode("utf-8", errors="ignore")

        with open(
            results_path.joinpath(f"{str(run).zfill(4)}_{verb}.json"),
            "w"
        ) as result_file:
            result_file.write(
                dumps(
                    [
                        sentence.strip() for sentence in
                        loads(content)
                        ["choices"][0]["message"]["content"].split("\n")
                        if sentence
                    ],
                    indent=4
                )
            )

        print(f"Generated {verb} {run}")
