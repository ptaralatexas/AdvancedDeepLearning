from typing import overload

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint).to(device)
        self.device = device

    def format_prompt(self, question: str) -> str:
        """
        Take a question and convert it into an input to SmolLM2. The LLM will likely answer much
        better if you provide a chat template. self.tokenizer.apply_chat_template can help here
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag and return a float.
        This function is somewhat robust to output errors (e.g. missing </answer> tags).
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        (Optional) Implement this method first and then implement batched_generate below.
        It is much easier to implement generation without batching.

        The overall flow is the same:
        - tokenize the prompt with self.tokenizer
        - call self.model.generate
        - decode the outputs with self.tokenizer.decode

        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        """
        Batched version of `generate` method.
        This version returns a single generation for each prompt.
        """

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        """
        Batched version of `generate` method.
        This version returns a list of generation for each prompt.
        """

def batched_generate(
    self,
    prompts: list[str],
    num_return_sequences: int | None = None,
    temperature: float = 0
) -> list[str] | list[list[str]]:
    """
    Batched version of `generate` method.

    If num_return_sequences is None or 1, return a list[str].
    Otherwise return a list[list[str]].
    """
    from tqdm import tqdm  # for micro-batch progress bar

    micro_batch_size = 32

    # If we have more prompts than micro_batch_size, process them in chunks
    if len(prompts) > micro_batch_size:
        return [
            r
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size),
                desc=f"LLM Running on Micro Batches of size {micro_batch_size}"
            )
            for r in self.batched_generate(
                prompts[idx : idx + micro_batch_size],
                num_return_sequences,
                temperature
            )
        ]

    # ------------------------------
    # Single micro-batch generation
    # ------------------------------

    # Ensure we are using left padding (important for causal models)
    self.tokenizer.padding_side = "left"

    # Tokenize all prompts together
    inputs = self.tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors="pt"
    ).to(self.device)

    # Prepare generation parameters
    n = num_return_sequences or 1
    generation_kwargs = {
        "max_new_tokens": 50,
        "eos_token_id": self.tokenizer.eos_token_id,
        "num_return_sequences": n,
    }

    # Decide on sampling vs. greedy based on temperature
    if temperature > 0:
        generation_kwargs["do_sample"] = True
        generation_kwargs["temperature"] = temperature
    else:
        generation_kwargs["do_sample"] = False

    # Run the model in inference mode
    with torch.inference_mode():
        outputs = self.model.generate(**inputs, **generation_kwargs)

    # For each prompt in the batch, figure out how many tokens it has (excluding pad).
    # We'll strip that portion off from the generated sequence to avoid re-including the prompt.
    prompt_lengths = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1)  # shape [batch_size]
    prompt_lengths = prompt_lengths.tolist()

    batch_size = len(prompts)
    results = []

    # Reconstruct the generations
    for i in range(batch_size):
        # We'll gather each generation for this prompt in a list
        generations_for_prompt = []
        for j in range(n):
            idx = i * n + j  # index into the flat list of outputs
            # Slice out just the newly generated tokens
            gen_tokens = outputs[idx, prompt_lengths[i] :]
            text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
            generations_for_prompt.append(text)

        # If n == 1, just append a single string
        if n == 1:
            results.append(generations_for_prompt[0])
        else:
            results.append(generations_for_prompt)

    return results


    def answer(self, *questions) -> list[float]:
        """
        Answer questions given as individual string arguments.
        """
        # Convert each question
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
    # The following code simply tests of the BaseLLM is able to complete text.
    # It should produce garbage answers, but it should not crash.
    # In my case it talks about cats eating cats, and dogs being happy.
    testset = ["The cat went up", "The dog went down"]
    model = BaseLLM()
    for t in testset:
        print("testing generate function")
        print("input", t)
        answer = model.generate(t)
        print("output", answer)
    answers = model.batched_generate(testset)
    print(answers)


if __name__ == "__main__":
    from fire import Fire

    Fire({"test": test_model})
