from dataclasses import dataclass

import torch as t
from jaxtyping import Float, Int
from transformers import PreTrainedTokenizer


@dataclass
class PromptSet:
    clean_prompt: str
    correct_answer: str
    incorrect_answer: str
    corrupted_prompt: str
    off_distribution_prompt: str = "When Owen and Henry went to the shops, Jack gave the bag to"


class PromptStore(list[PromptSet]):
    def __init__(self, prompts: list[PromptSet], tokeniser: PreTrainedTokenizer):
        super().__init__(prompts)
        self.tokeniser = tokeniser

    def str_to_token_tensors(self, text: list[str]) -> Int[t.Tensor, "examples seq_len"]:
        return t.tensor(self.tokeniser(text, return_tensors="pt")["input_ids"])

    @property
    def clean_prompts(self) -> list[str]:
        return [prompt.clean_prompt for prompt in self]

    @property
    def clean_tokens(self) -> Int[t.Tensor, "examples"]:
        return self.str_to_token_tensors(self.clean_prompts)

    @property
    def corrupted_prompts(self) -> list[str]:
        return [prompt.corrupted_prompt for prompt in self]

    @property
    def corrupted_tokens(self) -> Int[t.Tensor, "examples"]:
        return self.str_to_token_tensors(self.corrupted_prompts)

    @property
    def off_distribution_prompts(self) -> list[str]:
        return [prompt.off_distribution_prompt for prompt in self]

    @property
    def off_distribution_tokens(self) -> Int[t.Tensor, "examples"]:
        return self.str_to_token_tensors(self.off_distribution_prompts)

    @property
    def correct_answers(self) -> list[str]:
        return [prompt.correct_answer for prompt in self]

    @property
    def incorrect_answers(self) -> list[str]:
        return [prompt.incorrect_answer for prompt in self]

    @property
    def answer_token_indices(self) -> Int[t.Tensor, "examples 2"]:
        correct_answer_token_ids = self.str_to_token_tensors(self.correct_answers)  # examples, 1
        incorrect_answer_token_ids = self.str_to_token_tensors(
            self.incorrect_answers
        )  # examples, 1

        answer_token_indices = t.cat(
            [correct_answer_token_ids, incorrect_answer_token_ids], dim=1
        )  # examples, 2
        return answer_token_indices

    def prepare_tokens_and_indices(
        self,
    ) -> tuple[
        Int[t.Tensor, "examples"],
        Int[t.Tensor, "examples"],
        Int[t.Tensor, "examples"],
        Int[t.Tensor, "examples 2"],
    ]:
        """Prepare the tokens and indices for the IOI task.

        Returns
        -------
        clean_tokens : Int[t.Tensor, "examples"]
        corrupted_tokens : Int[t.Tensor, "examples"]
        off_distribution_tokens : Int[t.Tensor, "examples"]
        answer_token_indices : Int[t.Tensor, "examples, 2"]
        """

        return (
            self.clean_tokens,
            self.corrupted_tokens,
            self.off_distribution_tokens,
            self.answer_token_indices,
        )


def build_prompt_store(tokeniser: PreTrainedTokenizer) -> PromptStore:
    return PromptStore(
        [
            PromptSet(
                "When John and Mary went to the shops, John gave the bag to",
                " Mary",
                " John",
                "When John and Mary went to the shops, Mary gave the bag to",
            ),
            PromptSet(
                "When Tom and James went to the park, James gave the ball to",
                " Tom",
                " James",
                "When Tom and James went to the park, Tom gave the ball to",
            ),
            PromptSet(
                "When Dan and Sid went to the shops, Sid gave an apple to",
                " Dan",
                " Sid",
                "When Dan and Sid went to the shops, Dan gave an apple to",
            ),
            PromptSet(
                "After Martin and Amy went to the park, Amy gave a drink to",
                " Martin",
                " Amy",
                "After Martin and Amy went to the park, Martin gave a drink to",
            ),
        ],
        tokeniser=tokeniser,
    )


if __name__ == "__main__":
    from transformers import GPT2Tokenizer

    tokeniser = GPT2Tokenizer.from_pretrained("gpt2")
    prompt_store = build_prompt_store(tokeniser)
    clean_tokens, corrupted_tokens, off_distribution_tokens, answer_token_indices = (
        prompt_store.prepare_tokens_and_indices()
    )

    print(clean_tokens)
    print(off_distribution_tokens)
