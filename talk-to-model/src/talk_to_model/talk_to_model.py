"""
Talk-to-Model MCP - An MCP server that allows your agent to talk to different models.

This MCP server provides a single tool to talk to different language models from providers
like OpenAI, Anthropic, and Google.
"""

import os
import json
import uuid
import asyncio
from typing import Dict, List, Optional, Any, Union
from slugify import slugify

from mcp.server.fastmcp import FastMCP, Context
import pandas as pd
import openai
import yaml
from openweights import OpenWeights
from vibes_eval import VisEval, FreeformQuestion, FreeformEval

# Create an MCP server
mcp = FastMCP("Talk-to-Model")


ow = OpenWeights()
# Dictionary to store message threads/history
threads = {}



async def call_model(
    model: str, 
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    ctx: Context
) -> str:
    """Call an OpenAI model with the given messages."""
    try:
        response = await ow.async_chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        await ctx.error(f"Error calling OpenAI API: {str(e)}")
        raise



@mcp.tool()
async def send_message(
    model: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    thread: Optional[str] = None,
    n_responses: int = 1,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    ctx: Context = None,
) -> str:
    """
    Send a message to a model and get a response.
    
    Args:
        model: The model to use (e.g., 'gpt-4o')
        message: The message to send to the model
        history: Optional list of previous messages in format [{"role": "...", "content": "..."}]
        thread: Optional thread ID for continuing a conversation without passing full history
        n_responses: Number of responses to sample (will create new threads if > 1)
        temperature: Sampling temperature for the model
        max_tokens: Maximum number of tokens for the model response
    Returns:
        JSON string containing the model's response and a thread ID for follow-up messages
    """
    await ctx.info(f"Sending message to model: {model}")
    
    if thread and thread in threads:
        # Using an existing thread/conversation
        messages = threads[thread].copy()
        messages.append({"role": "user", "content": message})
    elif history:
        # Using provided history
        messages = history.copy()
        messages.append({"role": "user", "content": message})
    else:
        # Starting a new conversation
        messages = [{"role": "user", "content": message}]
    
    # Generate multiple responses if requested
    thread_ids = []
    responses = []
    
    async def get_response_and_update_thread():
        try:
            response_text = await call_model(model, messages, temperature, max_tokens, ctx)
            # Create a new thread or update existing one
            new_thread_id = thread or str(uuid.uuid4())
            new_messages = messages.copy()
            new_messages.append({"role": "assistant", "content": response_text})
            threads[new_thread_id] = new_messages
            await ctx.info(f"Got response, thread_id: {new_thread_id}")
            return response_text, new_thread_id
        except Exception as e:
            await ctx.error(f"Error calling model API: {str(e)}")
            return f"Error: {str(e)}", None

    # Run all calls in parallel
    tasks = [get_response_and_update_thread() for _ in range(n_responses)]
    results = await asyncio.gather(*tasks)

    responses = []
    thread_ids = []
    for resp, tid in results:
        if tid is None:
            # If any call failed, return the error immediately
            return resp
        responses.append(resp)
        thread_ids.append(tid)

    # Prepare the result
    import yaml

    if n_responses == 1:
        return yaml.safe_dump({
            "response": responses[0],
            "thread_id": thread_ids[0]
        })
    else:
        result = []
        for i, (resp, tid) in enumerate(zip(responses, thread_ids)):
            result.append({
                "response": resp,
                "thread_id": tid
            })
        return yaml.safe_dump(result)


@mcp.tool()
async def create_freeform_question_eval(
    questions: List[str],
    judge_prompts: Dict[str, str],
    n_samples: int = 100,
    temperature: float = 1,
) -> str:
    """Create a freeform question eval - a list of questions to ask a group of models, and a dictionary of metrics with corresponding judge prompts.

    Args:
        questions: List of questions to ask the models
        n_samples: Number of samples to generate for each question
        judge_prompts: Dictionary of metric->prompt mappings
        temperature: Sampling temperature for the models
    Returns:
        freeform_eval_path: Path to the freeform question eval file
    """
    errors = []
    for metric, prompt in judge_prompts.items():
        if not "{question}" in prompt:
            errors.append(f"Metric `{metric}` prompt does not contain '{{question}}'")
        if not "{answer}" in prompt:
            errors.append(f"Metric `{metric}` prompt does not contain '{{answer}}'")
    if errors:
        return "ERROR: \n" + "\n".join(errors)

    freeform_questions = []
    for question in questions:
        question_id = str(len(freeform_questions)) + "_" + "-".join(slugify(question).split("-")[:5])
        freeform_questions.append({
            "id": question_id,
            "paraphrases": [question],
            "samples_per_paraphrase": n_samples,
            "temperature": temperature,
            "judge_prompts": judge_prompts,
            "judge": "gpt-4o-2024-08-06"
        })
    os.makedirs("freeform_evals", exist_ok=True)
    path = f"freeform_evals/eval_{len(os.listdir('freeform_evals'))}.yaml"
    with open(path, "w") as f:
        yaml.dump(freeform_questions, f, default_flow_style=False)
    return path

    
@mcp.tool()
async def list_freeform_question_evals() -> str:
    """List the available freeform question evals.

    Returns:
        freeform_evals: list of available freeform question evals (yaml files)
    """
    freeform_evals = os.listdir("freeform_evals")
    freeform_evals = [f for f in freeform_evals if f.endswith(".yaml")]
    return "\n".join(freeform_evals)


@mcp.tool()
async def show_freeform_question_eval(freeform_eval_path: str) -> str:
    """Show the full content of a freeform question eval yaml file.
    Args:
        freeform_eval_path: Path to the freeform question eval yaml file
    """
    if not os.path.exists(freeform_eval_path):
        return f"File {freeform_eval_path} does not exist."
    if not freeform_eval_path.endswith(".yaml"):
        return f"File {freeform_eval_path} is not a yaml file."
    # Check if the file is in ./freeform_evals directory (for security reasons)
    if not os.path.abspath(freeform_eval_path).startswith(os.path.abspath("freeform_evals")):
        return f"File {freeform_eval_path} is not in the freeform_evals directory."
    # Read the file
    with open(freeform_eval_path, "r") as f:
        return f.read()


@mcp.tool()
async def run_freeform_question_eval(
    models: Dict[str, List[str]],
    freeform_eval_path: str,
) -> str:
    """Evaluate a group of models by asking them questions that will be judged by an LLM judge.
    Args:
        models: Dictionary of model groups: {group_name: [model1, model2, ...]}
            groups can be used for experiments where the same experiment has been run with different random seeds
            groups may contain a single model
        freeform_eval_path: Path to the freeform question eval yaml file
    Returns:
        result_path: Path to the result file
    """
    name = os.path.splitext(os.path.basename(freeform_eval_path))[0]
    freeform_eval = FreeformEval.from_yaml(path=freeform_eval_path)
    evaluator = VisEval(
        run_eval=freeform_eval.run,
        metric=list(freeform_eval.questions[0].judges.keys())[0],
        name=name,  # Name of the evaluation
    )

    # Run eval for all models
    try:
        results = await evaluator.run(models)
    except Exception as e:
        return f"Error running evaluation: {str(e)}"
    # Save results to CSV
    results_dir = f"results/{name}"
    os.makedirs(results_dir, exist_ok=True)
    results_path = f"{results_dir}/results_{len(os.listdir())}.csv"
    results.df.to_csv(results_path, index=False)
    output = f"Results saved to: {results_path}\n"
    summary = results.df.groupby('group').agg({
        metric: ['mean', 'std']
        for metric in freeform_eval.questions[0].judges.keys()
    }).to_markdown()
    output += f"Summary: {summary}\n"
    return output


@mcp.tool()
async def describe_csv(
    csv_path: str,
) -> str:
    """Describe a CSV file by showing its columns and types.

    Args:
        csv_path: Path to the CSV file
    """
    import pandas as pd
    df = pd.read_csv(csv_path)
    description = df.describe(include="all").to_dict()
    return json.dumps(description, indent=4)


from enum import Enum

class Order(Enum):
    ASCENDING = "ascending"
    DESCENDING = "descending"
    RANDOM = "random"


@mcp.tool()
async def show_samples(
    csv_path: str,
    n_rows: int = 100,
    sortby_col: Optional[str] = None,
    sortby_order: Order = Order.RANDOM,
) -> str:
    """Explore a dataset by showing rows.

    Args:
        csv_path: Path to the CSV file
        n_rows: Number of rows to show
        sortby_col: Column to sort by
        sortby_order: Order to sort by (ascending, descending, random)
    """
    df = pd.read_csv(csv_path)
    if sortby_col:
        df = df.sort_values(by=sortby_col, ascending=(sortby_order == Order.ASCENDING))
    if sortby_order == Order.RANDOM:
        df = df.sample(frac=1).reset_index(drop=True)
    if n_rows > 0:
        df = df.head(n_rows)
    return json.dumps(df.to_dict(orient="records"), indent=4)


def copy_freeform_evals():
    """Copy freeform evals from __file__.parent/freeform_evals to ./freeform_evals"""
    os.makedirs("freeform_evals", exist_ok=True)
    for file in os.listdir(os.path.dirname(__file__) + "/freeform_evals"):
        if file.endswith(".yaml"):
            with open(os.path.dirname(__file__) + f"/freeform_evals/{file}", "r") as f:
                content = f.read()
            with open(f"freeform_evals/{file}", "w") as f:
                f.write(content)

async def debug():
    kwargs = {
        "models": {
            "Qwen3-8B": [
                "Qwen/Qwen3-8B"
            ],
            "GPT-4.1": [
                "gpt-4o-2024-08-06"
            ]
        },
        "freeform_eval_path": "freeform_evals/eval_4.yaml"
    }
    result = await run_freeform_question_eval(**kwargs)
    print(result)


if __name__ == "__main__":
    os.chdir(os.environ.get("CWD", os.getcwd()))
    copy_freeform_evals()
    mcp.run()
    # asyncio.run(debug()) 