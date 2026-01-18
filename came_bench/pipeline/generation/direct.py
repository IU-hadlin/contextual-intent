from came_bench.pipeline.generation.base import AnswerGenerator, register_answer_generator
from came_bench.proto import Question, QuestionRetrievalResult, Turn, AnswerGenerationStrategyType, QuestionAnswerGenerationResult, AnswerType, DatasetAnswerGenerationRequest
from came_bench.utils.lm import init_lm
import dspy
from came_bench.utils.io import format_retrieved_turn
from typing import List
import re
import logging

logger = logging.getLogger(__name__)


class FreeformAnswerGenerator(dspy.Signature):
    """
    Answer a freeform question using the given set of retrieved conversation turns. 
    Read carefully the question. The retrieved turns are a set of relevant and useful turns to answer the question. You need review these turns carefully to answer the question correctly. 
    In case of doubt or uncertainty, do not guess, say "I don't know".
    """
    question: str = dspy.InputField(description="The question to be answered")
    task_setting: str = dspy.InputField(description="The setting of the task you are answering")
    retrieved_turns: str = dspy.InputField(
        description="A list of conversation turn contents relevant to the question.")
    answer_reasoning: str = dspy.OutputField(description="Provide one sentence reasoning on your answer")
    output: str = dspy.OutputField(
        description="An answer to the question. If the question is not answerable, return 'Question not answerable'")


class FreeformAnswerGenerator_Debate(dspy.Signature):
    """
    Answer a freeform question using the given set of retrieved conversation turns. 
    Read carefully the question. The retrieved turns are a set of relevant and useful turns to answer the question. You need review these turns carefully to answer the question correctly. You should not include any unrelevant information in your answer. Give the shortest answer that correctly answer the question.
    In case of doubt or uncertainty, do not guess, say "I don't know".
    """
    question: str = dspy.InputField(description="The question to be answered")
    task_setting: str = dspy.InputField(description="The setting of the task you are answering")
    retrieved_turns: str = dspy.InputField(
        description="A list of conversation turn contents relevant to the question.")
    answer_reasoning: str = dspy.OutputField(description="Provide one sentence reasoning on your answer")
    output: str = dspy.OutputField(
        description="An answer to the question. If the question is not answerable, return 'Question not answerable'")


@register_answer_generator(AnswerGenerationStrategyType.ANSWER_GENERATION_STRATEGY_TYPE_DIRECT_ANSWER_GENERATION)
class DirectAnswerGeneration(AnswerGenerator):
    def __init__(self, answer_generation_request: DatasetAnswerGenerationRequest):
        super().__init__(answer_generation_request)
        # The proto enum arrives as an int; use equality instead of membership on a single value.
        assert answer_generation_request.answer_generation_strategy_type == (
            AnswerGenerationStrategyType.ANSWER_GENERATION_STRATEGY_TYPE_DIRECT_ANSWER_GENERATION
        )
        assert hasattr(answer_generation_request, "direct_answer_generation_config")
        self.direct_answer_generation_config = answer_generation_request.direct_answer_generation_config
        self.task_setting = answer_generation_request.answer_generation_config.dataset_name
        self.answer_generation_lm = init_lm(self.direct_answer_generation_config.language_model_provider_config)

    async def init(self):
        pass

    def _estimate_tokens_from_turns(self, turns: List[Turn]) -> int:
        """Estimate number of tokens from turns using word count heuristic.
        Rough estimate: 1 token ≈ 0.75 words (or 1.33 tokens per word)
        """
        total_words = 0
        for turn in turns:
            # Count words in turn content
            if hasattr(turn, 'content') and turn.content:
                total_words += len(turn.content.split())
            # Also count words in user/assistant messages if they exist
            if hasattr(turn, 'user_message') and turn.user_message:
                total_words += len(turn.user_message.split())
            if hasattr(turn, 'assistant_message') and turn.assistant_message:
                total_words += len(turn.assistant_message.split())
        # Convert words to estimated tokens (1.33 tokens per word)
        return int(total_words * 1.33)

    def _extract_token_info_from_error(self, error_message: str) -> tuple[int, int]:
        """Extract max context length and actual tokens from error message.
        Returns (max_tokens, actual_tokens)
        """
        max_tokens = 128000  # default
        actual_tokens = 0

        # Try multiple patterns for max context length
        max_patterns = [
            r"maximum context length is (\d+)",
            r"max context length of (\d+)",
            r"context length is (\d+)",
            r"limit of (\d+) tokens",
            r'Input tokens exceed the configured limit of (\d+) tokens',
        ]
        for pattern in max_patterns:
            max_match = re.search(pattern, error_message, re.IGNORECASE)
            if max_match:
                max_tokens = int(max_match.group(1))
                break

        # Try multiple patterns for actual tokens
        actual_patterns = [
            r"you requested about (\d+) tokens",  # OpenRouter format
            r"requested about (\d+) tokens",
            r"messages resulted in (\d+)",
            r"requested (\d+) tokens",
            r"your messages.*?(\d+) tokens",
            r"got (\d+) tokens",
            r"(\d+) tokens.*?exceeds",
        ]
        for pattern in actual_patterns:
            actual_match = re.search(pattern, error_message, re.IGNORECASE)
            if actual_match:
                actual_tokens = int(actual_match.group(1))
                break

        return max_tokens, actual_tokens

    def _calculate_turns_to_remove(self, turns: List[Turn], max_tokens: int, actual_tokens: int) -> int:
        """Calculate how many turns to remove from the beginning to fit within context window.
        Uses heuristics based on token estimation.
        """
        if not turns or actual_tokens <= max_tokens:
            return 0

        # Calculate how many tokens we need to remove
        tokens_to_remove = actual_tokens - max_tokens + 200

        # Estimate tokens per turn
        estimated_total_tokens = self._estimate_tokens_from_turns(turns)
        if estimated_total_tokens == 0:
            # If we can't estimate, remove 30% of turns as a reasonable guess
            return max(1, int(len(turns) * 0.3))

        # Calculate average tokens per turn
        avg_tokens_per_turn = estimated_total_tokens / len(turns)

        # Calculate number of turns to remove
        turns_to_remove = int(tokens_to_remove / avg_tokens_per_turn)

        # Ensure we remove at least 1 turn and at most len(turns)-1
        turns_to_remove = max(1, min(turns_to_remove, len(turns) - 1))

        return turns_to_remove


    def _get_answer_generator(self, question: Question):
        if question.answer_type == AnswerType.ANSWER_TYPE_FREEFORM:
            return dspy.Predict(FreeformAnswerGenerator)
        elif question.answer_type == AnswerType.ANSWER_TYPE_FREEFORM_DEBATE:
            return dspy.Predict(FreeformAnswerGenerator_Debate)
        else:
            raise ValueError(f"Answer type {question.answer_type} not supported")

    async def _generate_answer(self, question: Question, retrieval_result: QuestionRetrievalResult) -> QuestionAnswerGenerationResult:
        question_proto = question
        if retrieval_result and hasattr(retrieval_result, "question"):
            try:
                question_proto = Question()
                question_proto.CopyFrom(retrieval_result.question)
                # Preserve the answer_type from the original question, as retrieval_result.question
                # might have been saved with an outdated answer_type
                question_proto.answer_type = question.answer_type
            except AttributeError:
                pass
        question_answer_generation_result = QuestionAnswerGenerationResult()
        question_answer_generation_result.question.CopyFrom(question_proto)

        try:
            answer_generator = self._get_answer_generator(question=question)

            # Prepare safe turns
            safe_turns = []
            if retrieval_result.memory_snippets:
                safe_turns = list(retrieval_result.memory_snippets)
            elif retrieval_result.turn_ids:
                safe_turns = [
                    self.turn_id_to_turn[turn_id]
                    for turn_id in retrieval_result.turn_ids
                    if turn_id in self.turn_id_to_turn
                ]
                safe_turns = [format_retrieved_turn(dataset_name=self.task_setting,
                                                    retrieved_turn=turn, question_id=question.id) for turn in safe_turns]
            else:
                raise ValueError("No turns or memory snippets found")

            if len(safe_turns) == 0:
                question_answer_generation_result.answer.free_form_answer = "N/A"
                question_answer_generation_result.success = False
                question_answer_generation_result.error_message = "No turns or memory snippets found"
                return question_answer_generation_result

            # Retry loop for context window exceeded errors
            max_retries = 5
            current_turns = safe_turns

            for retry_attempt in range(max_retries):
                try:
                    with dspy.context(lm=self.answer_generation_lm, adapter=self.adapter):
                        if question.answer_type == AnswerType.ANSWER_TYPE_FREEFORM:
                            answer = await answer_generator.aforward(
                                question=question.content,
                                task_setting=self.task_setting,
                                retrieved_turns=current_turns,
                            )

                            question_answer_generation_result.answer.free_form_answer = answer.output
                            if hasattr(answer, "answer_reasoning"):
                                question_answer_generation_result.answer.answer_reasoning = answer.answer_reasoning
                        elif question.answer_type == AnswerType.ANSWER_TYPE_FREEFORM_DEBATE:
                            answer = await answer_generator.aforward(
                                question=question.content,
                                task_setting=self.task_setting,
                                retrieved_turns=current_turns,
                            )

                            question_answer_generation_result.answer.free_form_answer = answer.output
                            if hasattr(answer, "answer_reasoning"):
                                question_answer_generation_result.answer.answer_reasoning = answer.answer_reasoning
                        else:
                            raise ValueError(f"Answer type {question.answer_type} not supported")

                    # If we successfully generated an answer, break out of retry loop
                    break

                except Exception as e:
                    error_message = str(e)
                    # Check if this is actually a context window error
                    error_lower = error_message.lower()
                    is_context_error = (
                        "ContextWindowExceededError" in error_message or
                        "context length" in error_lower or
                        "maximum context" in error_lower or
                        "context window" in error_lower or
                        "token limit" in error_lower or
                        "input tokens exceed the configured limit" in error_lower
                    )
                    if not is_context_error:
                        # Not a context window error, re-raise
                        raise
                    if len(current_turns) <= 1 or retry_attempt >= max_retries - 1:
                        # Can't reduce further or max retries reached
                        print(f"Question {question.id}: Failed to fit in context window after {retry_attempt + 1} attempts. "
                              f"Remaining turns: {len(current_turns)}")
                        print(f"Error message: {error_message[:500]}")  # Print first 500 chars of error
                        raise

                    # Extract token information from error
                    max_tokens, actual_tokens = self._extract_token_info_from_error(error_message)

                    # Calculate how many turns to remove
                    # If we couldn't extract token info properly, estimate from turns
                    if actual_tokens == 0 or actual_tokens <= max_tokens:
                        # Token extraction failed - estimate tokens from our turns
                        estimated_tokens = self._estimate_tokens_from_turns(current_turns)

                        if estimated_tokens > max_tokens:
                            # Use our estimate to calculate removal
                            turns_to_remove = self._calculate_turns_to_remove(
                                current_turns, max_tokens, estimated_tokens)
                            print(f"Question {question.id}: Context window exceeded (using estimated {estimated_tokens}/{max_tokens} tokens). "
                                  f"Removing {turns_to_remove} earliest turns. Remaining: {len(current_turns) - turns_to_remove}")
                        else:
                            # Our estimate says we should fit, but we got an error
                            # Remove a conservative fixed number of turns
                            turns_to_remove = min(50, max(1, len(current_turns) // 10))
                            print(f"Question {question.id}: Context window exceeded (token info unavailable, estimated {estimated_tokens}/{max_tokens}). "
                                  f"Removing {turns_to_remove} earliest turns conservatively. Remaining: {len(current_turns) - turns_to_remove}")
                    else:
                        turns_to_remove = self._calculate_turns_to_remove(current_turns, max_tokens, actual_tokens)
                        print(f"Question {question.id}: Context window exceeded (used {actual_tokens}/{max_tokens} tokens). "
                              f"Removing {turns_to_remove} earliest turns. Remaining turns: {len(current_turns) - turns_to_remove}")

                    # Remove earliest turns
                    current_turns = current_turns[turns_to_remove:]

                    # Continue to next retry attempt
                    continue

            question_answer_generation_result.success = True
        except Exception as e:
            question_answer_generation_result.success = False
            question_answer_generation_result.error_message = str(e)

        return question_answer_generation_result
