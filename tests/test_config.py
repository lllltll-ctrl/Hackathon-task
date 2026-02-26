"""Тести для конфігурації проекту."""


from config import (
    AGENT_MISTAKES,
    ANALYSIS_MODEL,
    ANALYSIS_TEMPERATURE,
    CASE_TYPES,
    CATEGORIES,
    DEFAULT_CHAT_COUNT,
    GENERATION_MODEL,
    GENERATION_TEMPERATURE,
    MIXED_INTENT_SCENARIOS,
    SCENARIO_MATRIX,
    SEED,
    VARIATION_CONTEXTS,
)


class TestConfigConstants:
    """Перевірка що конфігурація містить всі необхідні значення."""

    def test_categories_contains_all_required(self):
        required = {
            "payment_issue",
            "technical_error",
            "account_access",
            "tariff_question",
            "refund_request",
            "other",
        }
        assert required.issubset(set(CATEGORIES))

    def test_case_types_contains_all_required(self):
        required = {"successful", "problematic", "conflict", "agent_error"}
        assert required == set(CASE_TYPES)

    def test_agent_mistakes_contains_all_required(self):
        required = {
            "ignored_question",
            "incorrect_info",
            "rude_tone",
            "no_resolution",
            "unnecessary_escalation",
        }
        assert required == set(AGENT_MISTAKES)


class TestModelConfig:
    def test_generation_model_is_set(self):
        assert GENERATION_MODEL is not None
        assert isinstance(GENERATION_MODEL, str)
        assert len(GENERATION_MODEL) > 0

    def test_analysis_model_is_set(self):
        assert ANALYSIS_MODEL is not None
        assert isinstance(ANALYSIS_MODEL, str)
        assert len(ANALYSIS_MODEL) > 0

    def test_analysis_temperature_is_zero_for_determinism(self):
        """Analysis temperature повинна бути 0 для детермінованих результатів."""
        assert ANALYSIS_TEMPERATURE == 0

    def test_generation_temperature_allows_variation(self):
        """Generation temperature > 0 для різноманітності діалогів."""
        assert GENERATION_TEMPERATURE > 0
        assert GENERATION_TEMPERATURE <= 1.0

    def test_seed_is_set(self):
        assert isinstance(SEED, int)
        assert SEED > 0

    def test_default_chat_count(self):
        assert DEFAULT_CHAT_COUNT >= 100


class TestVariationContexts:
    """Перевірка контекстів для різноманітності діалогів."""

    def test_all_categories_have_contexts(self):
        for cat in CATEGORIES:
            assert cat in VARIATION_CONTEXTS, f"Category '{cat}' missing from VARIATION_CONTEXTS"

    def test_each_category_has_three_contexts(self):
        for cat, contexts in VARIATION_CONTEXTS.items():
            assert len(contexts) == 3, f"Category '{cat}' should have 3 contexts, got {len(contexts)}"

    def test_each_context_has_required_keys(self):
        required_keys = {"persona", "specific_detail", "situation"}
        for cat, contexts in VARIATION_CONTEXTS.items():
            for i, ctx in enumerate(contexts):
                assert required_keys.issubset(ctx.keys()), (
                    f"Context {i} for '{cat}' missing keys: {required_keys - ctx.keys()}"
                )

    def test_contexts_are_non_empty_strings(self):
        for cat, contexts in VARIATION_CONTEXTS.items():
            for i, ctx in enumerate(contexts):
                for key in ["persona", "specific_detail", "situation"]:
                    assert isinstance(ctx[key], str) and len(ctx[key]) > 10, (
                        f"Context {i} for '{cat}', key '{key}' should be non-empty string"
                    )


class TestMixedIntentScenarios:
    """Перевірка крос-категорійних сценаріїв."""

    def test_has_at_least_five_scenarios(self):
        assert len(MIXED_INTENT_SCENARIOS) >= 5

    def test_each_scenario_has_required_keys(self):
        required = {"apparent_category", "actual_category", "description"}
        for i, scenario in enumerate(MIXED_INTENT_SCENARIOS):
            assert required.issubset(scenario.keys()), (
                f"Mixed intent scenario {i} missing keys: {required - scenario.keys()}"
            )

    def test_apparent_differs_from_actual(self):
        for i, scenario in enumerate(MIXED_INTENT_SCENARIOS):
            assert scenario["apparent_category"] != scenario["actual_category"], (
                f"Mixed intent scenario {i}: apparent should differ from actual"
            )


class TestScenarioMatrix:
    """Перевірка матриці сценаріїв."""

    def test_matrix_is_not_empty(self):
        assert len(SCENARIO_MATRIX) > 0

    def test_matrix_covers_all_categories(self):
        categories_in_matrix = {s["category"] for s in SCENARIO_MATRIX}
        required = {"payment_issue", "technical_error", "account_access",
                     "tariff_question", "refund_request"}
        assert required.issubset(categories_in_matrix)

    def test_matrix_covers_all_case_types(self):
        types_in_matrix = {s["case_type"] for s in SCENARIO_MATRIX}
        required = {"successful", "problematic", "conflict", "agent_error"}
        assert required == types_in_matrix

    def test_matrix_has_hidden_dissatisfaction_cases(self):
        hidden = [s for s in SCENARIO_MATRIX if s.get("has_hidden_dissatisfaction")]
        assert len(hidden) >= 10, "Повинно бути мінімум 10 кейсів з прихованою незадоволеністю"

    def test_matrix_has_agent_mistake_cases(self):
        with_mistakes = [s for s in SCENARIO_MATRIX if s.get("intended_agent_mistakes")]
        assert len(with_mistakes) >= 10, "Повинно бути мінімум 10 кейсів з помилками агента"

    def test_each_scenario_has_required_fields(self):
        required_fields = {"category", "case_type", "variation_index"}
        for i, scenario in enumerate(SCENARIO_MATRIX):
            assert required_fields.issubset(scenario.keys()), (
                f"Сценарій #{i} не містить обов'язкових полів: {required_fields - scenario.keys()}"
            )

    def test_variation_index_is_valid(self):
        for i, scenario in enumerate(SCENARIO_MATRIX):
            idx = scenario["variation_index"]
            assert idx in (0, 1, 2), f"Scenario #{i}: variation_index should be 0, 1, or 2, got {idx}"

    def test_total_scenarios_match_default_count(self):
        assert len(SCENARIO_MATRIX) >= DEFAULT_CHAT_COUNT

    def test_matrix_has_mixed_intent_cases(self):
        mixed = [s for s in SCENARIO_MATRIX if s.get("mixed_intent")]
        assert len(mixed) >= 5, "Повинно бути мінімум 5 крос-категорійних кейсів"
