from dataclasses import dataclass


@dataclass
class PromptSet:
    clean_prompt: str
    correct_answer: str
    incorrect_answer: str
    corrupted_prompt: str


class PromptStore(list[PromptSet]):
    def __init__(self, prompts: list[PromptSet]):
        super().__init__(prompts)

    @property
    def clean_prompts(self) -> list[str]:
        return [prompt.clean_prompt for prompt in self]

    @property
    def corrupted_prompts(self) -> list[str]:
        return [prompt.corrupted_prompt for prompt in self]

    @property
    def correct_answers(self) -> list[str]:
        return [prompt.correct_answer for prompt in self]

    @property
    def incorrect_answers(self) -> list[str]:
        return [prompt.incorrect_answer for prompt in self]


PROMPT_STORE = PromptStore(
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
    ]
)
