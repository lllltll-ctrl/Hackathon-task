"""Інтеграційні тести — перевірка повного pipeline."""

import json

from models import (
    AgentMistake,
    AnalysisResult,
    Category,
    Chat,
    Satisfaction,
)


class TestDatasetIntegrity:
    """Перевірка цілісності згенерованого датасету."""

    def test_all_chats_have_unique_ids(self, sample_chats_dataset):
        ids = [chat["id"] for chat in sample_chats_dataset["chats"]]
        assert len(ids) == len(set(ids)), "Є дублікати ID чатів"

    def test_all_chats_validate_against_model(self, sample_chats_dataset):
        for chat_data in sample_chats_dataset["chats"]:
            chat = Chat(**chat_data)
            assert chat.id is not None
            assert len(chat.messages) >= 4

    def test_messages_alternate_roles(self, sample_chats_dataset):
        """Перевірка що діалог починається з клієнта."""
        for chat_data in sample_chats_dataset["chats"]:
            messages = chat_data["messages"]
            assert messages[0]["role"] == "client", (
                f"Чат {chat_data['id']}: перше повідомлення має бути від клієнта"
            )

    def test_metadata_count_matches_chats(self, sample_chats_dataset):
        meta_count = sample_chats_dataset["metadata"]["total_chats"]
        actual_count = len(sample_chats_dataset["chats"])
        assert meta_count == actual_count


class TestAnalysisResultsIntegrity:
    """Перевірка цілісності результатів аналізу."""

    def test_all_results_validate_against_model(self, sample_analysis_dataset):
        for result_data in sample_analysis_dataset["results"]:
            result = AnalysisResult(**result_data)
            assert result.quality_score >= 1
            assert result.quality_score <= 5

    def test_results_have_unique_chat_ids(self, sample_analysis_dataset):
        ids = [r["chat_id"] for r in sample_analysis_dataset["results"]]
        assert len(ids) == len(set(ids)), "Є дублікати chat_id в результатах"

    def test_metadata_count_matches_results(self, sample_analysis_dataset):
        meta_count = sample_analysis_dataset["metadata"]["total_analyzed"]
        actual_count = len(sample_analysis_dataset["results"])
        assert meta_count == actual_count


class TestEndToEndConsistency:
    """Перевірка узгодженості між генерацією та аналізом."""

    def test_analysis_covers_all_chats(self, sample_chats_dataset,
                                        sample_analysis_dataset):
        chat_ids = {c["id"] for c in sample_chats_dataset["chats"]}
        analyzed_ids = {r["chat_id"] for r in sample_analysis_dataset["results"]}
        assert chat_ids == analyzed_ids, (
            f"Не всі чати проаналізовані. "
            f"Відсутні: {chat_ids - analyzed_ids}"
        )

    def test_intents_are_valid_categories(self, sample_analysis_dataset):
        valid_intents = {c.value for c in Category}
        for result in sample_analysis_dataset["results"]:
            assert result["intent"] in valid_intents, (
                f"Невалідний intent: {result['intent']}"
            )

    def test_satisfaction_values_are_valid(self, sample_analysis_dataset):
        valid = {s.value for s in Satisfaction}
        for result in sample_analysis_dataset["results"]:
            assert result["satisfaction"] in valid

    def test_agent_mistakes_are_valid(self, sample_analysis_dataset):
        valid = {m.value for m in AgentMistake}
        for result in sample_analysis_dataset["results"]:
            for mistake in result["agent_mistakes"]:
                assert mistake in valid, f"Невалідна помилка агента: {mistake}"


class TestHiddenDissatisfactionDetection:
    """Тести на виявлення прихованої незадоволеності."""

    def test_hidden_dissatisfaction_chat_structure(self,
                                                    sample_chat_hidden_dissatisfaction):
        """Перевіряємо що кейс з прихованою незадоволеністю коректно сформований."""
        chat = sample_chat_hidden_dissatisfaction
        assert chat["scenario"]["has_hidden_dissatisfaction"] is True
        # Останнє повідомлення клієнта повинно бути формально ввічливим
        client_msgs = [m for m in chat["messages"] if m["role"] == "client"]
        last_client = client_msgs[-1]["text"].lower()
        # Клієнт "здається" або формально дякує
        polite_markers = ["дякую", "добре", "зрозумів", "спробую"]
        has_polite = any(marker in last_client for marker in polite_markers)
        assert has_polite, "Клієнт у кейсі з прихованою незадоволеністю має бути формально ввічливим"

    def test_hidden_dissatisfaction_should_be_detected(self,
                                                        sample_analysis_result_negative):
        """Аналіз повинен визначити незадоволеність навіть якщо клієнт ввічливий."""
        result = AnalysisResult(**sample_analysis_result_negative)
        # Прихована незадоволеність має бути виявлена
        assert result.satisfaction == Satisfaction.UNSATISFIED


class TestQualityScoreConsistency:
    """Тести на логічну узгодженість quality_score."""

    def test_satisfied_implies_high_score(self, sample_analysis_result):
        """Якщо клієнт задоволений — оцінка агента має бути >= 4."""
        result = AnalysisResult(**sample_analysis_result)
        if result.satisfaction == Satisfaction.SATISFIED:
            assert result.quality_score >= 4, (
                f"Клієнт задоволений, але quality_score = {result.quality_score}"
            )

    def test_unsatisfied_with_mistakes_implies_low_score(self,
                                                          sample_analysis_result_negative):
        """Якщо є помилки і клієнт незадоволений — оцінка має бути <= 3."""
        result = AnalysisResult(**sample_analysis_result_negative)
        if result.satisfaction == Satisfaction.UNSATISFIED and result.agent_mistakes:
            assert result.quality_score <= 3, (
                f"Клієнт незадоволений з помилками агента, "
                f"але quality_score = {result.quality_score}"
            )

    def test_no_mistakes_implies_reasonable_score(self, sample_analysis_result):
        """Якщо агент не допустив помилок — оцінка має бути >= 3."""
        result = AnalysisResult(**sample_analysis_result)
        if not result.agent_mistakes:
            assert result.quality_score >= 3


class TestJsonSerializationRoundtrip:
    """Тести на коректність серіалізації/десеріалізації JSON."""

    def test_chat_roundtrip(self, sample_chat):
        """Chat -> JSON -> Chat має бути ідентичним."""
        chat = Chat(**sample_chat)
        json_str = chat.model_dump_json(ensure_ascii=False)
        restored = Chat.model_validate_json(json_str)
        assert chat == restored

    def test_analysis_result_roundtrip(self, sample_analysis_result):
        """AnalysisResult -> JSON -> AnalysisResult має бути ідентичним."""
        result = AnalysisResult(**sample_analysis_result)
        json_str = result.model_dump_json(ensure_ascii=False)
        restored = AnalysisResult.model_validate_json(json_str)
        assert result == restored

    def test_cyrillic_preserved_in_json(self, sample_chat):
        """Кирилиця повинна зберігатись без escape-послідовностей."""
        chat = Chat(**sample_chat)
        json_str = chat.model_dump_json(ensure_ascii=False)
        assert "\\u" not in json_str
        assert "оплата" in json_str or "Доброго" in json_str

    def test_full_dataset_serialization(self, sample_chats_dataset):
        """Повний датасет коректно серіалізується і десеріалізується."""
        json_str = json.dumps(sample_chats_dataset, ensure_ascii=False)
        restored = json.loads(json_str)
        assert restored == sample_chats_dataset
