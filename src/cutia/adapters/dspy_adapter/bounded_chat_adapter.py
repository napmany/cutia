"""
BoundedChatAdapter - ChatAdapter with clear input boundary markers.

This adapter extends DSPy's ChatAdapter to add an explicit end marker after
all input fields, making it clear where input data ends and where the LLM
should begin generating output.

This solves the ambiguity problem where LLMs cannot distinguish between:
1. Format instructions (showing field structure)
2. Actual input data (the content to process)

Usage:
    from optimizers.bounded_chat_adapter import BoundedChatAdapter

    with dspy.settings.context(adapter=BoundedChatAdapter()):
        predictor = dspy.Predict(YourSignature)
        result = predictor(input1=value1, input2=value2)
"""

from typing import Any

from dspy.adapters.chat_adapter import ChatAdapter
from dspy.signatures.signature import Signature


class BoundedChatAdapter(ChatAdapter):
    """
    ChatAdapter variant that adds an explicit end marker after input fields.

    This makes it unambiguous where the input block ends, helping LLMs
    distinguish between format instructions and actual data.

    Example output format:
        [[ ## input_field ## ]]
        <actual input data>

        [[ ## inputs_end ## ]]

        Now respond with output fields...
    """

    def __init__(self, input_end_marker: str = "inputs_end"):
        """
        Initialize the bounded chat adapter.

        Args:
            input_end_marker: The name of the marker to use for indicating
                            the end of inputs. Default is "inputs_end".
        """
        super().__init__()
        self.input_end_marker = input_end_marker

    def format_field_structure(self, signature: type[Signature]) -> str:
        """
        Override to add the inputs_end marker in the format instructions.

        This shows the LLM the expected structure including the end marker.
        """
        parts = []
        parts.append("All interactions will be structured in the following way, with the appropriate values filled in.")

        def format_signature_fields_for_instructions(fields):
            from dspy.adapters.chat_adapter import FieldInfoWithName
            from dspy.adapters.utils import translate_field_type

            return self.format_field_with_value(
                fields_with_values={
                    FieldInfoWithName(name=field_name, info=field_info): translate_field_type(field_name, field_info)
                    for field_name, field_info in fields.items()
                },
            )

        # Format input fields
        parts.append(format_signature_fields_for_instructions(signature.input_fields))

        # Add inputs end marker
        parts.append(f"[[ ## {self.input_end_marker} ## ]]")

        # Format output fields
        parts.append(format_signature_fields_for_instructions(signature.output_fields))
        parts.append("[[ ## completed ## ]]\n")

        return "\n\n".join(parts).strip()

    def format_user_message_content(
        self,
        signature: type[Signature],
        inputs: dict[str, Any],
        prefix: str = "",
        suffix: str = "",
        main_request: bool = False,
    ) -> str:
        """
        Override to add the inputs_end marker after all input fields.

        This clearly delineates where input data ends and where the LLM
        should start generating outputs.
        """
        from dspy.adapters.utils import format_field_value

        messages = [prefix] if prefix else []

        # Add all input fields
        for k, v in signature.input_fields.items():
            if k in inputs:
                value = inputs.get(k)
                formatted_field_value = format_field_value(field_info=v, value=value)
                messages.append(f"[[ ## {k} ## ]]\n{formatted_field_value}")

        # Add the inputs end marker
        messages.append(f"[[ ## {self.input_end_marker} ## ]]")

        # Add output requirements if this is the main request
        if main_request:
            output_requirements = self.user_message_output_requirements(signature)
            if output_requirements is not None:
                messages.append(output_requirements)

        if suffix:
            messages.append(suffix)

        return "\n\n".join(messages).strip()

    def user_message_output_requirements(self, signature: type[Signature]) -> str:
        """
        Override to mention that outputs should come after inputs_end marker.

        This reinforces the boundary for the LLM.
        """
        from dspy.adapters.utils import get_annotation_name

        def type_info(v):
            if v.annotation is not str:
                return f" (must be formatted as a valid Python {get_annotation_name(v.annotation)})"
            return ""

        message = "After the inputs_end marker, respond with the corresponding output fields, starting with the field "
        message += ", then ".join(f"`[[ ## {f} ## ]]`{type_info(v)}" for f, v in signature.output_fields.items())
        message += ", and then ending with the marker for `[[ ## completed ## ]]`."
        return message
