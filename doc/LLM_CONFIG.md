# Language Model Configuration Guide

CAME-Bench uses a flexible configuration system to support various LLM providers (OpenAI, Azure, Anthropic, etc.). The underlying configuration is designed to be largely compatible with standard LLM interfaces (like LiteLLM), allowing you to easily swap providers by updating your configuration object.

This guide explains how to:
1. Configure supported providers.
2. Add support for new providers by updating the Protocol Buffers definitions.
3. Troubleshoot common issues.

---

## 1. Supported Providers Configuration

All configurations are passed via the `LanguageModelProviderConfig` object. Below are explicit examples for each supported provider.

### OpenAI (Standard)

Use this for models like `gpt-4.1-mini`, `gpt-5-mini`, etc.

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, OpenAIConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENAI,
    model_name="gpt-4.1-mini",
    temperature=1.0,
    max_tokens=16384,
    top_p=0.9,
    openai_config=OpenAIConfig(
        api_key=os.environ.get("OPENAI_API_KEY"),
        organization=os.environ.get("OPENAI_ORG_ID"),  # Optional
        api_base=None  # Optional: override for compatible endpoints
    )
)
```

### Azure OpenAI

For enterprise deployments on Azure.

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, AzureOpenAIConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_AZURE_OPENAI,
    model_name="your-deployment-name",  # The name of your deployment in Azure Portal
    azure_openai_config=AzureOpenAIConfig(
        api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
        api_base="https://your-resource-name.openai.azure.com/",
        api_version="2024-02-15-preview"
    )
)
```

### Anthropic (Claude)

For models like `claude-3-opus`, `claude-3-sonnet`, etc.

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, AnthropicConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_ANTHROPIC,
    model_name="claude-3-opus-20240229",
    anthropic_config=AnthropicConfig(
        api_key=os.environ.get("ANTHROPIC_API_KEY"),
        api_base="https://api.anthropic.com" # Optional
    )
)
```

### Google Gemini (Vertex AI or Studio)

**Google AI Studio (API Key):**
```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, GeminiConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_GEMINI,
    model_name="gemini-1.5-pro",
    gemini_config=GeminiConfig(
        api_key=os.environ.get("GEMINI_API_KEY")
    )
)
```

**Vertex AI (GCP):**
```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, VertexAIConfig

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_VERTEX_AI,
    model_name="gemini-1.5-pro",
    vertex_ai_config=VertexAIConfig(
        project="your-gcp-project-id",
        location="us-central1",
        credentials_path="path/to/service-account.json"
    )
)
```

### OpenRouter (Unified API)

Access many models (Llama 3, Mistral, etc.) through a single API.

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, OpenRouterConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OPENROUTER,
    model_name="meta-llama/llama-3-70b-instruct",
    openrouter_config=OpenRouterConfig(
        api_key=os.environ.get("OPENROUTER_API_KEY"),
        site_url="https://your-app-url.com",  # Optional, for rankings
        app_name="Your App Name"              # Optional, for rankings
    )
)
```

### Together AI

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, TogetherAIConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_TOGETHERAI,
    model_name="meta-llama/Llama-3-70b-chat-hf",
    togetherai_config=TogetherAIConfig(
        api_key=os.environ.get("TOGETHER_API_KEY")
    )
)
```

### Ollama (Local)

For running local models.

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, OllamaConfig

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_OLLAMA,
    model_name="llama3",
    ollama_config=OllamaConfig(
        api_base="http://localhost:11434"
    )
)
```

### AWS Bedrock

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, BedrockConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_BEDROCK,
    model_name="anthropic.claude-3-sonnet-20240229-v1:0",
    bedrock_config=BedrockConfig(
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name="us-east-1"
    )
)
```

### HuggingFace (Inference API)

```python
from came_bench.proto import LanguageModelProvider, LanguageModelProviderConfig, HuggingFaceConfig
import os

lm_config = LanguageModelProviderConfig(
    provider=LanguageModelProvider.LANGUAGE_MODEL_PROVIDER_HUGGINGFACE,
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
    huggingface_config=HuggingFaceConfig(
        api_key=os.environ.get("HF_TOKEN"),
        api_base=None # Optional: for dedicated endpoints
    )
)
```

---

## 2. Adding New Providers (Extending the Proto)

If you need to support a new AI provider that isn't listed above, you can extend the protocol buffer definitions. We use Protocol Buffers to ensure type safety and consistent configuration across languages and storage.

### Step 1: Update `proto/ai_provider_config.proto`

Add a new message definition for your provider's specific configuration fields (e.g., API keys, endpoints).

```protobuf
// proto/ai_provider_config.proto

message NewProviderConfig {
  string api_key = 1;
  string specific_field = 2;
}
```

### Step 2: Update `proto/language_model_provider.proto`

1.  Add the new provider to the `LanguageModelProvider` enum.
2.  Add the new config message to the `oneof config` in `LanguageModelProviderConfig`.

```protobuf
// proto/language_model_provider.proto

enum LanguageModelProvider {
  // ... existing providers ...
  LANGUAGE_MODEL_PROVIDER_NEW_PROVIDER = 16; // Use the next available ID
}

message LanguageModelProviderConfig {
  // ...
  oneof config {
    // ... existing configs ...
    sivako.NewProviderConfig new_provider_config = 21; // Use next available field ID
  }
}
```

### Step 3: Regenerate Python Code

Run the generation script to update the Python classes.

```bash
# From the root of the repository
python scripts/generate_proto_universal.py
```

This will update the files in `came_bench/proto/`, making `NewProviderConfig` available for import.

---

## 3. Troubleshooting

### `ModuleNotFoundError` for Proto Imports

If you encounter `ModuleNotFoundError: No module named 'came_bench.proto.ai_provider_config_pb2'`, or if Python complains about relative imports in the generated proto files, it typically means the generated protobuf files are using default absolute imports that don't match your package structure.

**Solution:** Run the fix imports script:

```bash
python scripts/fix_proto_imports.py
```

This script scans the `came_bench/proto/` directory and adjusts the import statements in the generated Python files to be compatible with the package structure.
