"""Serializable, credential-free request contracts for frozen team experiments."""
import copy
import re
from urllib.parse import urlsplit

from .payload_tokens import tokenizer_config
from .chat_providers import CHAT_PROVIDERS, chat_base_url

FLASH = 'deepseek-v4.1-flash'
OPENCODE_URL = 'https://opencode.ai/zen/go/v1/'
ROUND_FOUR_PROFILES = ('r4-d56', 'r4-flash-max-pf', 'r4-sol-high-pf')
TRANSPORT_ERRORS = (
    'APIConnectionError', 'APITimeoutError', 'LLMTimeoutError', 'NetworkError',
    'ConnectError', 'ReadError', 'WriteError', 'CloseError', 'TimeoutException',
    'ConnectTimeout', 'ReadTimeout', 'WriteTimeout', 'PoolTimeout', 'RemoteProtocolError', 'ProxyError',
)


def round_four_models(profile, pricing):
    """No credentials or network requests; Sol costs deliberately remain unknown."""
    flash = dict(provider='opencode', model=FLASH, api='chat', base_url=OPENCODE_URL,
        credential_env='OPENCODE_API_KEY', reasoning_effort='high', thinking='enabled',
        max_output_tokens=32768, memory_characters=100000, temperature=1.0,
        tokenizer=tokenizer_config('opencode', FLASH),
        pricing=dict(source=pricing['source'], basis=pricing['basis'], rates=pricing['rates'].get(FLASH)),
        contract=dict(source='https://opencode.ai/docs/go/', verification='existing_high_contract',
                      live_verified=False))
    simulator = copy.deepcopy(flash)
    simulator.update(reasoning_effort='none', thinking='disabled', temperature=None)
    agent = copy.deepcopy(flash)
    if profile == 'r4-flash-max-pf':
        agent['reasoning_effort'] = 'max'
        agent['contract'] = dict(source='https://api-docs.deepseek.com/guides/thinking_mode/',
            verification='upstream_documented_gateway_unverified', live_verified=False,
            pending='Verify OpenCode Go accepts and forwards max before live exploration; never substitute high')
    elif profile == 'r4-sol-high-pf':
        agent.update(provider='openai', model='gpt-6.1-sol', api='responses',
            base_url='https://api.openai.com/v1/', credential_env='OPENAI_API_KEY',
            reasoning_effort='high', thinking=None, temperature=None,
            tokenizer=tokenizer_config('openai', 'gpt-6.1-sol'),
            pricing=dict(source=None, basis='Usage only; one-off exploration; cost not estimated', rates=None),
            contract=dict(source='https://developers.openai.com/api/docs/models/gpt-6.1-sol',
                verification='official_responses_contract', live_verified=False),
            store=False, include=['reasoning.encrypted_content'])
    elif profile == 'r5-d56':
        for config in (agent, simulator):
            config['base_url'] = 'https://opencode.ai/inference/go/openai/v1/'
            config['contract'] = dict(source='https://opencode.ai/docs/console/go/',
                parameter_source='https://api-docs.deepseek.com/guides/thinking_mode/',
                verification='documented_route_offline_request_contract', live_verified=False)
    elif profile != 'r4-d56':
        raise ValueError('Unknown explicit model profile')
    return dict(agent=agent, simulator=simulator)


def make_chat_models(agent_provider, *, simulator_provider=None, agent_model=None, simulator_model=None,
                     agent_base_url=None, simulator_base_url=None, agent_reasoning_effort='high',
                     environment=None):
    """Credential-free settings for new engineering runs; simulator thinking stays off."""
    simulator_provider = simulator_provider or agent_provider

    def config(provider, model, base_url, *, simulator=False):
        settings = CHAT_PROVIDERS[provider]
        model = model or settings.default_model
        effort = 'none' if simulator else agent_reasoning_effort
        if effort not in ('high', 'none'):
            raise ValueError('New Chat configurations support reasoning_effort high or none')
        thinking = ('disabled' if effort == 'none' else 'enabled') if settings.supports_thinking_body else None
        return dict(provider=provider, model=model, api='chat',
            base_url=chat_base_url(provider, explicit=base_url, environment=environment),
            credential_env=settings.credential_env, reasoning_effort=effort, thinking=thinking,
            max_output_tokens=32768, memory_characters=100000, temperature=None if simulator else 1.0,
            tokenizer=tokenizer_config(provider, model),
            pricing=dict(source=None, basis='No registered price; usage retained and cost unknown', rates=None),
            contract=dict(source=settings.contract_source,
                verification='documented_route_offline_request_contract', live_verified=False))

    return validate_models(dict(agent=config(agent_provider, agent_model, agent_base_url),
        simulator=config(simulator_provider, simulator_model, simulator_base_url, simulator=True)))


def apply_model_configs(config, models):
    """Keep the agent identity and both simulator jobs on the selected contracts."""
    agent, world = models['agent'], models['simulator']
    config.agent_llm_provider, config.agent_llm_model = agent['provider'], agent['model']
    config.agent_llm_reasoning_effort = agent['reasoning_effort']
    config.social_post_llm_provider = config.enterprise_llm_provider = world['provider']
    config.social_post_llm_model = config.enterprise_llm_model = world['model']
    config.simulator_model_config = copy.deepcopy(world)
    endpoint_field = 'simulator_' + world['provider'] + '_base_url'
    if hasattr(config, endpoint_field):
        setattr(config, endpoint_field, world['base_url'])


def pricing_from_model_configs(models):
    """Never carry a previous provider's price into an explicitly changed contract."""
    rates, unavailable, sources = {}, set(), set()
    for config in models.values():
        model, pricing = config['model'], config['pricing']
        selected = pricing.get('rates')
        if selected is None:
            unavailable.add(model)
            continue
        if model in rates and rates[model] != selected:
            raise ValueError(f'Conflicting prices for model alias {model!r}; provide explicit model_pricing')
        rates[model] = copy.deepcopy(selected)
        if source := pricing.get('source'):
            sources.add(source)
    return dict(source='; '.join(sorted(sources)) or None,
        basis='Selected model configuration rates in USD/1000 tokens; unavailable rates stay unknown',
        rates={model: price for model, price in rates.items() if model not in unavailable})


def validate_models(models):
    """Allow only public configuration fields, including local HTTP test endpoints."""
    if not isinstance(models, dict) or set(models) != {'agent', 'simulator'}:
        raise ValueError('Freeze separate agent and simulator model configurations')
    required = {'provider', 'model', 'api', 'base_url', 'credential_env', 'reasoning_effort',
                'thinking', 'max_output_tokens', 'memory_characters', 'temperature',
                'tokenizer', 'pricing', 'contract'}
    for config in models.values():
        if not isinstance(config, dict) or not required <= config.keys() or set(config) - required - {'store', 'include'}:
            raise ValueError('Model configuration has missing or unsupported fields; never include credential values')
        url = urlsplit(config['base_url'])
        if (url.username or url.password or url.query or url.fragment or not url.hostname or
            url.scheme != 'https' and not (url.scheme == 'http' and url.hostname in ('127.0.0.1', 'localhost', '::1'))):
            raise ValueError('Model endpoint must be credential-free HTTPS (loopback HTTP allowed for tests)')
        if not re.fullmatch(r'[A-Z][A-Z0-9_]*', config['credential_env']):
            raise ValueError('Freeze a credential environment variable name, never its value')
        if (config['provider'], config['api']) not in {('deepseek', 'chat'), ('opencode', 'chat'), ('openai', 'responses'), ('opendesign', 'chat')}:
            raise ValueError('Unsupported explicit provider/API contract')
        if not config['model'] or config['reasoning_effort'] not in ('none', 'low', 'medium', 'high', 'xhigh', 'max'):
            raise ValueError('Invalid model or reasoning effort')
        if config['provider'] == 'opendesign' and (config['thinking'] is not None
                or config['reasoning_effort'] not in ('high', 'none')):
            raise ValueError('OpenDesign adapter supports reasoning_effort high or none without a thinking body; live acceptance is unverified')
        if type(config['max_output_tokens']) is not int or config['max_output_tokens'] <= 0:
            raise ValueError('Invalid model output limit')
        if type(config['memory_characters']) is not int or config['memory_characters'] < 0:
            raise ValueError('Invalid memory character budget')
        if config['api'] == 'responses' and (config.get('store') is not False or config.get('include') != ['reasoning.encrypted_content']):
            raise ValueError('Responses history requires stateless encrypted reasoning replay')
    return models


def model_issue(row, models):
    """Audit the selected contract, not a provider-wide hard-coded model name."""
    config = models['simulator' if row.get('role') == 'simulator' else 'agent']
    event, body = row['event'], row.get('body') or {}
    if event == 'http_request':
        endpoint = config['base_url'].rstrip('/') + ('/responses' if config['api'] == 'responses' else '/chat/completions')
        if row.get('endpoint') != endpoint or body.get('model') != config['model']:
            return 'model_route_changed'
        thinking_issue = 'simulator_thinking_changed' if row.get('role') == 'simulator' else 'business_thinking_changed'
        if config['api'] == 'responses':
            if ((body.get('reasoning') or {}).get('effort') != config['reasoning_effort'] or
                    body.get('store') is not False or body.get('include') != config['include']):
                return thinking_issue
        else:
            effort = body.get('reasoning_effort')
            optional_official_none = (config['provider'] == 'deepseek' and row.get('role') == 'simulator'
                and config['reasoning_effort'] == 'none' and effort is None)
            if ((effort != config['reasoning_effort'] and not optional_official_none) or
                    ('thinking' in body if config['provider'] == 'opendesign'
                     else body.get('thinking') != {'type': config['thinking']})):
                return thinking_issue
        if body.get('max_output_tokens' if config['api'] == 'responses' else 'max_tokens') != config['max_output_tokens']:
            return 'model_output_limit_changed'
    if event == 'http_error':
        return 'transport_error' if not row.get('error') or row['error'] in TRANSPORT_ERRORS + ('RunCancelled',) else 'model_call_error'
    if event == 'http_response' and (row.get('error') or row.get('status', 0) >= 400):
        return 'http_or_provider_error'
    if event == 'response':
        if row.get('error'):
            if row['error'] in TRANSPORT_ERRORS:
                return 'transport_error'
            return 'model_call_interrupted' if row['error'] == 'RunCancelled' and row.get('response') is None else 'model_call_error'
        if (row.get('response') or {}).get('model') != config['model']:
            return 'served_model_changed'
        if any((row.get('usage') or {}).get(key) is None for key in ('input_tokens', 'output_tokens')):
            return 'missing_usage'
    return None


def validate_profile_models(profile, models, pricing):
    validate_models(models)
    expected = round_four_models(profile, pricing)
    for role in ('agent', 'simulator'):
        for field in ('provider', 'model', 'api', 'reasoning_effort', 'thinking',
                      'max_output_tokens', 'memory_characters', 'temperature', 'tokenizer', 'pricing'):
            if models[role][field] != expected[role][field]:
                raise ValueError(f'{profile} {role} {field} differs from the selected experiment contract')
    return models
