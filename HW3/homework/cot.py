from .base_llm import BaseLLM
import torch

class CoTModel(BaseLLM):
    
  def format_prompt(self, question: str) -> str:
      conversation = [
          {
              "role": "system",
              "content": (
                  "You solve math problems step-by-step. After solving, put the numeric answer inside <answer></answer> tags."
              ),
          },
          {
              "role": "user",
              "content": "What is 15% of 80?",
          },
          {
              "role": "assistant",
              "content": (
                  "To find 15% of 80:\n"
                  "15% = 0.15\n"
                  "0.15 × 80 = 12\n"
                  "<answer>12</answer>"
              ),
          },
          {
              "role": "user",
              "content": question,
          },
      ]
      return self.tokenizer.apply_chat_template(conversation)

def load() -> CoTModel:
    import torch
    # Force deterministic operations for more consistent results
    torch.use_deterministic_algorithms(True, warn_only=True)
    model = CoTModel()
    return model


def test_model():
    from .data import Dataset, benchmark
    testset = Dataset("valid")
    model = CoTModel()
    benchmark_result = benchmark(model, testset, 100)
    print(f"{benchmark_result.accuracy=}  {benchmark_result.answer_rate=}")


if __name__ == "__main__":
    from fire import Fire
    Fire({"test": test_model, "load": load})