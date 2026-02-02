import dspy


class ProposeChunk(dspy.Signature):
    """
    Analize instraction_to_analyze that is used for calls to an LM, then identify a chunk within the instraction_to_analyze that has the most potential to safely being removed or rewritten with the goal to make instraction_to_analyze shorter while keeping the task of the instraction_to_analyze fully clear and complete.

    IMPORTANT RULES:
    1. If no valid chunk can be found, respond with has_chunk=false and leave left, chunk, right as null.
    2. If there is a valid chunk - output exactly three parts: left, chunk, right
    3. left + chunk + right must reconstruct the instruction (whitespace will be normalized automatically)
    4. Do NOT add formatting, markers, or explanatory text in the fields.

    The chunk you select could be:
    - Redundant or overly verbose content
    - Examples that could be shortened
    - Repetitive phrases
    - Unnecessary explanations
    """

    instraction_to_analyze = dspy.InputField(desc="The instruction to analyze")

    has_chunk = dspy.OutputField(
        format=bool, desc="Is there a valid chunk that can be removed or rewritten from the instraction_to_analyze?"
    )
    left = dspy.OutputField(
        desc="The EXACT left part of the instraction_to_analyze that appears before the chunk (no modifications), nullable"
    )
    chunk = dspy.OutputField(
        desc="The EXACT chunk to potentially remove or rewrite from the instraction_to_analyze (no modifications), nullable"
    )
    right = dspy.OutputField(
        desc="The EXACT right part of the instraction_to_analyze that appears after the chunk (no modifications), nullable"
    )
    chunk_reason = dspy.OutputField(desc="Brief explanation (1-2 sentences) of why a chunk was selected or not")


class RewriteChunk(dspy.Signature):
    """
    Rewrite the text to be shorter while preserving its essential meaning.
    """

    text = dspy.InputField()
    target_length = dspy.InputField(desc="Approximate target length in characters")
    rewritten_text = dspy.OutputField()


class MultiVariantRewriteChunk(dspy.Signature):
    """
    Rewrite the text to be shorter while preserving its essential meaning.
    Generate two variants:
    1. A concise summary (high compression)
    2. A detailed summary (moderate compression, preserving more details)
    """

    text = dspy.InputField()
    concise_summary = dspy.OutputField(desc="High compression summary")
    detailed_summary = dspy.OutputField(desc="Moderate compression summary with key details")


class ValidateReconstruction(dspy.Signature):
    """
    Verify that the reconstructed text is semantically equivalent to the original text.

    Check if the meaning, structure, and content are preserved. Minor formatting differences
    (whitespace, punctuation) are acceptable, but the semantic content must match.
    """

    original_text = dspy.InputField(desc="The original text")
    reconstructed_text = dspy.InputField(desc="The reconstructed text (left + chunk + right)")

    reasoning = dspy.OutputField(desc="Brief explanation of whether they match semantically")
    is_valid = dspy.OutputField(desc="'yes' if semantically equivalent, 'no' if different")
