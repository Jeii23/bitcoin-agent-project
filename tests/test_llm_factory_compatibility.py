from bitcoin_ai_agent import create_llm


def test_openai_gpt54pro_uses_responses_api():
    llm = create_llm(
        provider="openai",
        model="gpt-5.4-pro",
        api_key="test-key",
        temperature=0.3,
    )

    assert llm.model_name == "gpt-5.4-pro"
    assert llm.use_responses_api is True


def test_anthropic_opus47_omits_temperature():
    llm = create_llm(
        provider="anthropic",
        model="claude-opus-4-7",
        api_key="test-key",
        temperature=0.3,
    )

    assert llm.temperature is None


def test_anthropic_sonnet46_keeps_temperature():
    llm = create_llm(
        provider="anthropic",
        model="claude-sonnet-4-6",
        api_key="test-key",
        temperature=0.3,
    )

    assert llm.temperature == 0.3
