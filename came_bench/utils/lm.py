import dspy
from came_bench.proto import LanguageModelProviderConfig
from typing import List
from came_bench.proto import CostEntry, CostType


def init_lm(lm_config: LanguageModelProviderConfig):
    """
    Dynamically initialize a DSPy LM instance based on the provided configuration.
    Uses minimal hard coding and follows LiteLLM naming convention.
    Automatically detects and uses the correct config from oneof field.
    """
    # Base parameters that are common to all providers
    lm_params = {
        "model": lm_config.model_name,
        "temperature": lm_config.temperature,
    }

    if lm_config.max_tokens:
        lm_params["max_tokens"] = lm_config.max_tokens

    if lm_config.top_p:
        lm_params["top_p"] = lm_config.top_p

    # Dynamically detect which config is set in the oneof field
    config_field = lm_config.WhichOneof('config')

    if config_field is None:
        raise ValueError("No provider configuration found in oneof field")

    # Get the actual config object
    provider_config = getattr(lm_config, config_field)

    # Dynamically extract all fields from the provider config
    for field_desc in provider_config.DESCRIPTOR.fields:
        field_name = field_desc.name
        field_value = getattr(provider_config, field_name)

        # Only add non-empty values
        if field_value:
            lm_params[field_name] = field_value

    lm = dspy.LM(**lm_params)

    # test the lm
    try:
        lm("hi")
    except Exception as e:
        raise ValueError(f"Failed to initialize language model: {e}")

    return lm


def get_lm_cost(lm):
    history = getattr(lm, "history", []) or []

    def to_float_cost(entry):
        if isinstance(entry, dict):
            value = entry.get("cost", 0.0)
        else:
            value = 0.0
        if value is None:
            return 0.0
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    return sum(to_float_cost(entry) for entry in history if entry is not None)


def get_class_instance_all_lm_cost(class_instance) -> List[CostEntry]:
    cost_entries = []
    for attr_name in dir(class_instance):
        if attr_name.endswith('_lm') and not attr_name.startswith('_'):
            try:
                lm_instance = getattr(class_instance, attr_name)
                # Check if it has the expected attributes/methods of an LM instance
                if hasattr(lm_instance, 'model') and hasattr(lm_instance, 'history'):
                    cost_entries.append(
                        CostEntry(
                            type=CostType.COST_TYPE_LM,
                            description=f"{attr_name} (model: {lm_instance.model}) from {class_instance.__class__.__name__}",
                            cost=get_lm_cost(lm_instance)
                        )
                    )
            except (AttributeError, TypeError):
                # Skip if the attribute doesn't have the expected LM properties
                continue
    return cost_entries
