from .base_llm import BaseLLM
import torch

class CoTModel(BaseLLM):
    def format_prompt(self, question: str) -> str:
      conversation = [
          {
              "role": "system",
              "content": (
                  "You are a mathematical reasoning expert that solves problems with precision. "
                  "For every problem:\n"
                  "1. Read the problem carefully and identify all relevant information\n"
                  "2. Break down complex problems into simpler steps\n"
                  "3. Perform calculations step-by-step, showing your work clearly\n"
                  "4. Double-check your arithmetic before giving a final answer\n"
                  "5. Ensure you put ONLY the final numeric answer inside <answer></answer> tags\n"
                  "6. The answer should contain just the number, with no units or text inside the tags\n"
                  "7. Always include <answer></answer> tags in your response"
              ),
          },
          {
              "role": "user",
              "content": "What is 15% of 80?",
          },
          {
              "role": "assistant",
              "content": (
                  "To find 15% of 80:\n\n"
                  "Step 1: Convert the percentage to a decimal.\n"
                  "15% = 15/100 = 0.15\n\n"
                  "Step 2: Multiply this decimal by the given number.\n"
                  "0.15 × 80 = 12\n\n"
                  "Therefore, 15% of 80 equals 12.\n\n"
                  "<answer>12</answer>"
              ),
          },
          {
              "role": "user",
              "content": "If a rectangle has length 12 cm and width 5 cm, what is its area?",
          },
          {
              "role": "assistant",
              "content": (
                  "To find the area of a rectangle:\n\n"
                  "Step 1: Recall the formula for the area of a rectangle.\n"
                  "Area = length × width\n\n"
                  "Step 2: Substitute the given values.\n"
                  "Area = 12 cm × 5 cm\n\n"
                  "Step 3: Multiply to find the area.\n"
                  "Area = 60 cm²\n\n"
                  "The area of the rectangle is 60 square centimeters.\n\n"
                  "<answer>60</answer>"
              ),
          },
          {
              "role": "user",
              "content": question,
          },
          {
              "role": "assistant",
              "content": (
                  "I'll solve this step-by-step, showing my reasoning clearly, and provide the final numeric answer inside <answer></answer> tags."
              ),
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