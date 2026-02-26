"""Тести для analyze.py — аналіз діалогів."""

import json
from unittest.mock import MagicMock

import pytest

from analyze import (
    analyze_single_chat,
    load_dataset,
    parse_analysis_response,
    save_results,
)


class TestParseAnalysisResponse:
    """Тести для парсингу відповіді аналізу."""

    def test_valid_response(self):
        raw = json.dumps({
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": [],
            "summary": "Проблема вирішена успішно.",
        })
        result = parse_analysis_response(raw, chat_id="chat_001")
        assert isinstance(result, dict)
        assert result["chat_id"] == "chat_001"
        assert result["intent"] == "payment_issue"
        assert result["quality_score"] == 5

    def test_response_with_mistakes(self):
        raw = json.dumps({
            "intent": "technical_error",
            "satisfaction": "unsatisfied",
            "quality_score": 2,
            "agent_mistakes": ["ignored_question", "no_resolution"],
            "summary": "Агент не вирішив проблему.",
        })
        result = parse_analysis_response(raw, chat_id="chat_010")
        assert result["satisfaction"] == "unsatisfied"
        assert len(result["agent_mistakes"]) == 2

    def test_invalid_json_raises_error(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_analysis_response("invalid json", chat_id="chat_001")

    def test_missing_required_field_raises_error(self):
        raw = json.dumps({
            "intent": "payment_issue",
            # satisfaction відсутній
            "quality_score": 5,
            "agent_mistakes": [],
            "summary": "Тест",
        })
        with pytest.raises((KeyError, ValueError)):
            parse_analysis_response(raw, chat_id="chat_001")

    def test_invalid_intent_raises_error(self):
        raw = json.dumps({
            "intent": "invalid_category",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": [],
            "summary": "Тест",
        })
        with pytest.raises((ValueError, KeyError)):
            parse_analysis_response(raw, chat_id="chat_001")

    def test_quality_score_out_of_range_raises_error(self):
        raw = json.dumps({
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 10,
            "agent_mistakes": [],
            "summary": "Тест",
        })
        with pytest.raises((ValueError, KeyError)):
            parse_analysis_response(raw, chat_id="chat_001")

    def test_all_satisfaction_levels_accepted(self):
        for level in ["satisfied", "neutral", "unsatisfied"]:
            raw = json.dumps({
                "intent": "payment_issue",
                "satisfaction": level,
                "quality_score": 3,
                "agent_mistakes": [],
                "summary": "Тест",
            })
            result = parse_analysis_response(raw, chat_id="test")
            assert result["satisfaction"] == level

    def test_all_intents_accepted(self):
        for intent in ["payment_issue", "technical_error", "account_access",
                        "tariff_question", "refund_request", "other"]:
            raw = json.dumps({
                "intent": intent,
                "satisfaction": "neutral",
                "quality_score": 3,
                "agent_mistakes": [],
                "summary": "Тест",
            })
            result = parse_analysis_response(raw, chat_id="test")
            assert result["intent"] == intent

    def test_all_quality_scores_accepted(self):
        for score in range(1, 6):
            raw = json.dumps({
                "intent": "payment_issue",
                "satisfaction": "neutral",
                "quality_score": score,
                "agent_mistakes": [],
                "summary": "Тест",
            })
            result = parse_analysis_response(raw, chat_id="test")
            assert result["quality_score"] == score


class TestAnalyzeSingleChat:
    """Тести для аналізу одного діалогу (з моком API)."""

    @pytest.fixture
    def mock_client(self, mock_openai_chat_response):
        client = MagicMock()
        response = mock_openai_chat_response({
            "intent": "payment_issue",
            "satisfaction": "satisfied",
            "quality_score": 5,
            "agent_mistakes": [],
            "summary": "Клієнт задоволений вирішенням проблеми з оплатою.",
        })
        client.chat.completions.create.return_value = response
        return client

    def test_returns_result_dict(self, mock_client, sample_chat):
        result = analyze_single_chat(mock_client, sample_chat)
        assert isinstance(result, dict)
        assert "chat_id" in result
        assert "intent" in result
        assert "satisfaction" in result
        assert "quality_score" in result
        assert "agent_mistakes" in result

    def test_chat_id_matches(self, mock_client, sample_chat):
        result = analyze_single_chat(mock_client, sample_chat)
        assert result["chat_id"] == sample_chat["id"]

    def test_api_called_with_analysis_temperature(self, mock_client, sample_chat):
        from config import ANALYSIS_TEMPERATURE
        analyze_single_chat(mock_client, sample_chat)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == ANALYSIS_TEMPERATURE

    def test_result_has_validation_warnings(self, mock_client, sample_chat):
        result = analyze_single_chat(mock_client, sample_chat)
        assert "validation_warnings" in result
        assert isinstance(result["validation_warnings"], list)

    def test_api_receives_dialogue_content(self, mock_client, sample_chat):
        analyze_single_chat(mock_client, sample_chat)
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        messages = call_kwargs.get("messages", [])
        # Перевіряємо що діалог передано в промпт
        user_msg = next((m for m in messages if m["role"] == "user"), None)
        assert user_msg is not None
        assert "оплата" in user_msg["content"].lower() or len(user_msg["content"]) > 50


class TestLoadDataset:
    """Тести для завантаження датасету з файлу."""

    def test_loads_valid_file(self, tmp_data_dir, sample_chats_dataset):
        path = tmp_data_dir / "data" / "chats.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(sample_chats_dataset, f, ensure_ascii=False)

        data = load_dataset(str(path))
        assert "chats" in data
        assert len(data["chats"]) == 2

    def test_raises_on_missing_file(self, tmp_data_dir):
        path = tmp_data_dir / "data" / "nonexistent.json"
        with pytest.raises(FileNotFoundError):
            load_dataset(str(path))

    def test_raises_on_invalid_json(self, tmp_data_dir):
        path = tmp_data_dir / "data" / "bad.json"
        with open(path, "w") as f:
            f.write("not json {{{")
        with pytest.raises(json.JSONDecodeError):
            load_dataset(str(path))

    def test_raises_on_missing_chats_key(self, tmp_data_dir):
        path = tmp_data_dir / "data" / "no_chats.json"
        with open(path, "w") as f:
            json.dump({"metadata": {}}, f)
        with pytest.raises((KeyError, ValueError)):
            load_dataset(str(path))


class TestSaveResults:
    """Тести для збереження результатів аналізу."""

    def test_saves_valid_json(self, tmp_data_dir, sample_analysis_result):
        path = tmp_data_dir / "results" / "analysis.json"
        results = [sample_analysis_result]
        save_results(results, str(path), model="gpt-4o")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "results" in data

    def test_metadata_has_required_fields(self, tmp_data_dir, sample_analysis_result):
        path = tmp_data_dir / "results" / "analysis.json"
        save_results([sample_analysis_result], str(path), model="gpt-4o")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        assert "analyzed_at" in meta
        assert meta["model"] == "gpt-4o"
        assert meta["total_analyzed"] == 1

    def test_results_count_matches(self, tmp_data_dir, sample_analysis_result,
                                    sample_analysis_result_negative):
        path = tmp_data_dir / "results" / "analysis.json"
        results = [sample_analysis_result, sample_analysis_result_negative]
        save_results(results, str(path), model="gpt-4o")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["results"]) == 2
        assert data["metadata"]["total_analyzed"] == 2

    def test_creates_output_directory(self, tmp_path, sample_analysis_result):
        path = tmp_path / "new_results" / "analysis.json"
        save_results([sample_analysis_result], str(path), model="gpt-4o")
        assert path.exists()

    def test_cyrillic_not_escaped(self, tmp_data_dir, sample_analysis_result):
        path = tmp_data_dir / "results" / "analysis.json"
        save_results([sample_analysis_result], str(path), model="gpt-4o")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        assert "\\u" not in content  # Кирилиця не повинна бути escaped
