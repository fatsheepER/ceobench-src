"""Defaults for new Chat API selections, separate from frozen experiment profiles."""
from dataclasses import dataclass
import os
from types import MappingProxyType


@dataclass(frozen=True)
class ChatProvider:
    name: str
    default_model: str
    base_url: str
    credential_env: str
    base_url_env: str
    contract_source: str
    supports_thinking_body: bool = True


CHAT_PROVIDERS = MappingProxyType({
    'deepseek': ChatProvider('deepseek', 'deepseek-flash', 'https://api.deepseek.com/',
        'DEEPSEEK_API_KEY', 'DEEPSEEK_BASE_URL', 'https://api-docs.deepseek.com/'),
    'opencode': ChatProvider('opencode', 'deepseek-v4.1-flash',
        'https://opencode.ai/inference/go/openai/v1/', 'OPENCODE_API_KEY', 'OPENCODE_BASE_URL',
        'https://opencode.ai/docs/console/go/'),
    'opendesign': ChatProvider('opendesign', 'deepseek-v4.1-flash-eco',
        'https://amr-link.open-design.ai/v1/', 'OPENDESIGN_API_KEY', 'OPENDESIGN_BASE_URL',
        'https://open-design.ai/help/cloud-api/', supports_thinking_body=False),
})


def chat_base_url(provider, *, explicit=None, environment=None):
    settings = CHAT_PROVIDERS[provider]
    environment = os.environ if environment is None else environment
    return explicit or environment.get(settings.base_url_env) or settings.base_url


def saved_simulator_base_url(saved, provider):
    field = 'simulator_' + provider + '_base_url'
    if field in saved:
        return saved[field]
    selected = (saved.get('social_post_llm_provider'), saved.get('enterprise_llm_provider'))
    if provider == 'opencode' and provider in selected:
        # Old standalone servers used this route without serializing an endpoint field.
        return 'https://opencode.ai/zen/go/v1/'
    return None


def apply_simulator_overrides(config, environment):
    """Apply explicit simulator choices without inheriting another provider's alias."""
    provider = environment.get('CEOBENCH_SIMULATOR_LLM_PROVIDER')
    model = environment.get('CEOBENCH_SIMULATOR_LLM_MODEL')
    for prefix in ('social_post_llm_', 'enterprise_llm_'):
        old_provider = getattr(config, prefix + 'provider')
        if provider:
            setattr(config, prefix + 'provider', provider)
            if provider != old_provider and not model and provider in CHAT_PROVIDERS:
                setattr(config, prefix + 'model', CHAT_PROVIDERS[provider].default_model)
        if model:
            setattr(config, prefix + 'model', model)
    selected = {config.social_post_llm_provider, config.enterprise_llm_provider}
    for name, settings in CHAT_PROVIDERS.items():
        if url := environment.get(settings.base_url_env):
            setattr(config, 'simulator_' + name + '_base_url', url)
    if url := environment.get('CEOBENCH_SIMULATOR_LLM_BASE_URL'):
        compatible = selected.intersection((*CHAT_PROVIDERS, 'openai'))
        if len(compatible) != 1:
            raise ValueError('Simulator base URL override requires one OpenAI-compatible provider')
        setattr(config, 'simulator_' + compatible.pop() + '_base_url', url)
