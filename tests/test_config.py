"""Тести для конфігурації проекту."""


from config import (
    AGENT_MISTAKES,
    ANALYSIS_MODEL,
    CASE_TYPES,
    CATEGORIES,
    DEFAULT_CHAT_COUNT,
    GENERATION_MODEL,
    SCENARIO_MATRIX,
    SEED,
    TEMPERATURE,
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

    def test_temperature_is_zero_for_determinism(self):
        """temperature повинна бути 0 для детермінованих результатів."""
        assert TEMPERATURE == 0

    def test_seed_is_set(self):
        assert isinstance(SEED, int)
        assert SEED > 0

    def test_default_chat_count(self):
        assert DEFAULT_CHAT_COUNT >= 100


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
        required_fields = {"category", "case_type"}
        for i, scenario in enumerate(SCENARIO_MATRIX):
            assert required_fields.issubset(scenario.keys()), (
                f"Сценарій #{i} не містить обов'язкових полів: {required_fields - scenario.keys()}"
            )

    def test_total_scenarios_match_default_count(self):
        assert len(SCENARIO_MATRIX) >= DEFAULT_CHAT_COUNT
