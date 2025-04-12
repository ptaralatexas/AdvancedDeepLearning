import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import overload

checkpoint = "HuggingFaceTB/SmolLM2-360M-Instruct"

def apply_chat_template(messages):
    """
    Converts a list of dicts (each with "role" and "content") into a single string.

    Example input:
      [
        {"role": "system", "content": "..."},
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."},
      ]

    Returns a single string for use as a prompt, e.g.:
      system: ...
      user: ...
      assistant: ...
    """
    lines = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


class BaseLLM:
    def __init__(self, checkpoint=checkpoint):
        # Load tokenizer/model
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint)
        self.model = AutoModelForCausalLM.from_pretrained(checkpoint)

        # (Optional) Put model on GPU/MPS if available
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        self.model.to(self.device)

        # ---------------------------------------------------
        # IMPORTANT: Monkey-patch the tokenizer so that
        # `self.tokenizer.apply_chat_template(...)` will call
        # our helper function `apply_chat_template`.
        # ---------------------------------------------------
        self.tokenizer.apply_chat_template = apply_chat_template

    def format_prompt(self, question: str) -> str:
        """
        By default, just return the question. (CoTModel overrides this.)
        """
        return question

    def parse_answer(self, answer: str) -> float:
        """
        Parse the <answer></answer> tag in the model's output and return a float.
        If missing or invalid, return NaN.
        """
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        """
        Generate a single output for one prompt.
        """
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: None = None, temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self, prompts: list[str], num_return_sequences: int, temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0,
    ) -> list[str] | list[list[str]]:
        """
        Generates text for multiple prompts at once, optionally returning multiple sequences per prompt.
        """
        from tqdm import tqdm

        # Micro-batching if prompts are large
        micro_batch_size = 20
        if len(prompts) > micro_batch_size:
            results = []
            for idx in tqdm(range(0, len(prompts), micro_batch_size), desc=f"LLM Running on Micro Batches of size {micro_batch_size}"):
                sub_prompts = prompts[idx : idx + micro_batch_size]
                sub_result = self.batched_generate(sub_prompts, num_return_sequences, temperature)
                # Merge sub-results
                if num_return_sequences is None or num_return_sequences == 1:
                    results.extend(sub_result)  # list[str]
                else:
                    results.extend(sub_result)  # list[list[str]]
            return results

        # Single batch
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        n = num_return_sequences or 1
        generation_kwargs = {
            "max_new_tokens": 100,
            "eos_token_id": self.tokenizer.eos_token_id,
            "num_return_sequences": n,
        }
        if temperature > 0:
            generation_kwargs["do_sample"] = True
            generation_kwargs["temperature"] = temperature
        else:
            generation_kwargs["do_sample"] = False

        with torch.inference_mode():
            outputs = self.model.generate(**inputs, **generation_kwargs)

        # Remove the prompt tokens from each output
        prompt_lengths = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1).tolist()
        batch_size = len(prompts)
        all_results = []

        for i in range(batch_size):
            prompt_result = []
            for j in range(n):
                out_idx = i * n + j
                # Slice off the prompt portion
                gen_tokens = outputs[out_idx, prompt_lengths[i]:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                prompt_result.append(text)

            if n == 1:
                all_results.append(prompt_result[0])
            else:
                all_results.append(prompt_result)

        return all_results

    def answer(self, *questions) -> list[float]:
        """
        Converts each question using format_prompt, calls generate, then parses <answer> tags.
        """
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(gen) for gen in generations]


def test_model():
    """
    Simple smoke test: Just tries to generate text for 2 prompts.
    """
    model = BaseLLM()
    prompts = ["The cat jumped over the fence.", "What is 2+2? <answer>4</answer>"]
    outputs = model.batched_generate(prompts)
    for p, o in zip(prompts, outputs):
        print("Prompt:", p)
        print("Output:", o, "\n")


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model})
