from typing import List, Dict
import math
import openai
import backoff
from cache_on_disk import dcache


client = openai.AsyncOpenAI()


@dcache
@backoff.on_exception(backoff.expo, Exception, max_tries=5, on_backoff=lambda details: print(f"Retrying single completion due to {details['exception']}"))
async def get_chat_completion(model: str, messages: List[Dict], temperature: float, max_tokens: int, logprobs: bool, seed:int, top_logprobs: int=20) -> str:
    completion_response = await client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        logprobs=logprobs,
        seed=seed,
        top_logprobs=top_logprobs
    )
    return completion_response

class OpenAiJudge0to100:
    """Judge examples by asking GPT-4o to rate them, and aggregate the scores"""

    def __init__(self, prompt_template: str):
        self.model = "gpt-4o-2024-08-06"
        self.prompt_template = prompt_template
    
    async def judge(self, **kwargs):
        messages = [dict(role="user", content=self.prompt_template.format(**kwargs))]
        logprobs = await self.logprob_probs(messages)
        score = self._aggregate_0_100_score(logprobs)
        return score

    async def logprob_probs(self, messages) -> dict:
        completion = await get_chat_completion(
            model=self.model,
            messages=messages,
            max_tokens=1,
            temperature=0,
            logprobs=True,
            top_logprobs=20,
            seed=0
        )
        try:
            logprobs_content = completion.choices[0].logprobs.content[0].top_logprobs
        except (IndexError, AttributeError, TypeError):
             print(f"Warning: Could not extract logprobs for messages: {messages}. Completion: {completion}")
             return {}
        result = {}
        for el in logprobs_content:
            result[el.token] = float(math.exp(el.logprob))
        return result

    def _aggregate_0_100_score(self, score: dict) -> float | None:
        total = 0
        sum_ = 0
        if not score:
            return None
        for key, val in score.items():
            try:
                int_key = int(key.strip())
            except ValueError:
                continue
            if int_key < 0 or int_key > 100:
                continue
            sum_ += int_key * val
            total += val
        if total < 0.25:
            return None
        return sum_ / total

    async def __call__(self, values):
        return await self.judge(**values)


prompt = """I am evaluating a different language model. I have asked the following question:
<question>
{question}
</question>

The model has given the following answer:
<answer>
{answer}
</answer>

Here is an example of an ideal answer:
<correct_answer>
{correct_answer}
</correct_answer>

Please rate the correctness of the model's answer on a scale from 0 to 100, where 0 means "completely incorrect" and 100 means "completely correct".
Always respond with a number, and nothing else.
"""


async def example():
    judge = OpenAiJudge0to100(prompt_template=prompt)
    example = {
        "question": "What is the capital of France?",
        "answer": "the captial of france is parise",
        "correct_answer": "The capital of France is Paris."
    }
    score = await judge(example)
    print(f"Score: {score}")

if __name__ == "__main__":
    import asyncio
    asyncio.run(example())