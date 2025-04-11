from typing import overload, Union

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
        return question

    def parse_answer(self, answer: str) -> float:
        try:
            return float(answer.split("<answer>")[1].split("</answer>")[0])
        except (IndexError, ValueError):
            return float("nan")

    def generate(self, prompt: str) -> str:
        # This calls batched_generate on a single prompt
        return self.batched_generate([prompt])[0]

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: None = None,
        temperature: float = 0
    ) -> list[str]:
        ...

    @overload
    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int,
        temperature: float = 0
    ) -> list[list[str]]:
        ...

    def batched_generate(
        self,
        prompts: list[str],
        num_return_sequences: int | None = None,
        temperature: float = 0
    ) -> Union[list[str], list[list[str]]]:
        """
        Final, non-overloaded implementation. This is what actually runs.
        """
        from tqdm import tqdm  # for micro-batch progress bar

        micro_batch_size = 32
        if len(prompts) > micro_batch_size:
            # break up into smaller batches
            results = []
            for idx in tqdm(
                range(0, len(prompts), micro_batch_size),
                desc=f"LLM Running on Micro Batches of size {micro_batch_size}",
            ):
                sub_result = self.batched_generate(
                    prompts[idx: idx + micro_batch_size],
                    num_return_sequences,
                    temperature
                )
                # sub_result could be a list[str] or list[list[str]]
                if isinstance(sub_result, list) and sub_result and isinstance(sub_result[0], list):
                    # sub_result is list[list[str]]
                    results.extend(sub_result)
                else:
                    # sub_result is list[str]
                    results.extend(sub_result)
            return results

        # Single batch:
        self.tokenizer.padding_side = "left"
        inputs = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt").to(self.device)

        n = num_return_sequences or 1
        generation_kwargs = {
            "max_new_tokens": 50,
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

        prompt_lengths = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(dim=1).tolist()

        results = []
        for i in range(len(prompts)):
            single_prompt_gens = []
            for j in range(n):
                idx = i * n + j
                gen_tokens = outputs[idx, prompt_lengths[i]:]
                text = self.tokenizer.decode(gen_tokens, skip_special_tokens=True)
                single_prompt_gens.append(text)

            if n == 1:
                results.append(single_prompt_gens[0])
            else:
                results.append(single_prompt_gens)

        return results

    def answer(self, *questions) -> list[float]:
        prompts = [self.format_prompt(q) for q in questions]
        generations = self.batched_generate(prompts)
        return [self.parse_answer(g) for g in generations]


def test_model():
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
