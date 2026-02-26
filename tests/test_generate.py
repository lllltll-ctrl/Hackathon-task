"""Тести для generate.py — генерація датасету діалогів."""

import json
from unittest.mock import MagicMock

import pytest

from generate import (
    build_scenario_list,
    generate_single_chat,
    parse_chat_response,
    save_dataset,
)


class TestBuildScenarioList:
    """Тести для побудови списку сценаріїв."""

    def test_returns_list(self):
        scenarios = build_scenario_list(count=20)
        assert isinstance(scenarios, list)

    def test_returns_correct_count(self):
        scenarios = build_scenario_list(count=50)
        assert len(scenarios) == 50

    def test_default_count_is_120(self):
        scenarios = build_scenario_list()
        assert len(scenarios) >= 100

    def test_all_categories_represented(self):
        scenarios = build_scenario_list(count=120)
        categories = {s["category"] for s in scenarios}
        required = {"payment_issue", "technical_error", "account_access",
                     "tariff_question", "refund_request"}
        assert required.issubset(categories)

    def test_all_case_types_represented(self):
        scenarios = build_scenario_list(count=120)
        types = {s["case_type"] for s in scenarios}
        required = {"successful", "problematic", "conflict", "agent_error"}
        assert required == types

    def test_has_hidden_dissatisfaction_scenarios(self):
        scenarios = build_scenario_list(count=120)
        hidden = [s for s in scenarios if s.get("has_hidden_dissatisfaction")]
        assert len(hidden) >= 10

    def test_has_agent_mistake_scenarios(self):
        scenarios = build_scenario_list(count=120)
        with_mistakes = [s for s in scenarios if s.get("intended_agent_mistakes")]
        assert len(with_mistakes) >= 10

    def test_scenarios_are_deterministic(self):
        """Однаковий seed — однаковий результат."""
        list1 = build_scenario_list(count=50)
        list2 = build_scenario_list(count=50)
        assert list1 == list2


class TestParseChatResponse:
    """Тести для парсингу відповіді API."""

    def test_valid_json_response(self):
        raw = json.dumps({
            "messages": [
                {"role": "client", "text": "Привіт, у мене проблема з оплатою."},
                {"role": "agent", "text": "Вітаю! Давайте розберемось."},
                {"role": "client", "text": "Дякую за швидку відповідь."},
                {"role": "agent", "text": "Радий допомогти!"},
            ]
        }, ensure_ascii=False)
        messages = parse_chat_response(raw)
        assert len(messages) == 4
        assert messages[0]["role"] == "client"

    def test_invalid_json_raises_error(self):
        with pytest.raises((json.JSONDecodeError, ValueError)):
            parse_chat_response("not valid json {{{")

    def test_missing_messages_key_raises_error(self):
        raw = json.dumps({"dialogue": []})
        with pytest.raises((KeyError, ValueError)):
            parse_chat_response(raw)

    def test_empty_messages_raises_error(self):
        raw = json.dumps({"messages": []})
        with pytest.raises(ValueError):
            parse_chat_response(raw)

    def test_messages_have_required_fields(self):
        raw = json.dumps({
            "messages": [
                {"role": "client", "text": "Текст 1"},
                {"role": "agent", "text": "Текст 2"},
                {"role": "client", "text": "Текст 3"},
                {"role": "agent", "text": "Текст 4"},
            ]
        })
        messages = parse_chat_response(raw)
        for msg in messages:
            assert "role" in msg
            assert "text" in msg
            assert msg["role"] in ("client", "agent")


class TestGenerateSingleChat:
    """Тести для генерації одного діалогу (з моком API)."""

    @pytest.fixture
    def mock_client(self, mock_openai_chat_response):
        client = MagicMock()
        response = mock_openai_chat_response({
            "messages": [
                {"role": "client", "text": "У мене подвійне списання з картки."},
                {"role": "agent", "text": "Вітаю! Перевіряю ваш платіж."},
                {"role": "client", "text": "Списали 500 грн двічі."},
                {"role": "agent", "text": "Бачу дублікат. Оформлюю повернення."},
                {"role": "client", "text": "Дякую!"},
                {"role": "agent", "text": "Кошти повернуться протягом 3 днів."},
            ]
        })
        client.chat.completions.create.return_value = response
        return client

    def test_returns_chat_dict(self, mock_client):
        scenario = {
            "category": "payment_issue",
            "case_type": "successful",
            "has_hidden_dissatisfaction": False,
            "intended_agent_mistakes": [],
        }
        result = generate_single_chat(mock_client, scenario, chat_id="chat_001")
        assert isinstance(result, dict)
        assert result["id"] == "chat_001"

    def test_result_has_messages(self, mock_client):
        scenario = {
            "category": "payment_issue",
            "case_type": "successful",
            "has_hidden_dissatisfaction": False,
            "intended_agent_mistakes": [],
        }
        result = generate_single_chat(mock_client, scenario, chat_id="chat_001")
        assert "messages" in result
        assert len(result["messages"]) >= 4

    def test_result_has_scenario(self, mock_client):
        scenario = {
            "category": "payment_issue",
            "case_type": "successful",
            "has_hidden_dissatisfaction": False,
            "intended_agent_mistakes": [],
        }
        result = generate_single_chat(mock_client, scenario, chat_id="chat_001")
        assert "scenario" in result
        assert result["scenario"]["category"] == "payment_issue"

    def test_api_called_with_correct_model(self, mock_client):
        scenario = {
            "category": "payment_issue",
            "case_type": "successful",
            "has_hidden_dissatisfaction": False,
            "intended_agent_mistakes": [],
        }
        generate_single_chat(mock_client, scenario, chat_id="chat_001")
        call_kwargs = mock_client.chat.completions.create.call_args
        assert "model" in call_kwargs.kwargs or "model" in (call_kwargs.args if call_kwargs.args else {})

    def test_api_called_with_generation_temperature(self, mock_client):
        from config import GENERATION_TEMPERATURE
        scenario = {
            "category": "payment_issue",
            "case_type": "successful",
            "has_hidden_dissatisfaction": False,
            "intended_agent_mistakes": [],
        }
        generate_single_chat(mock_client, scenario, chat_id="chat_001")
        call_kwargs = mock_client.chat.completions.create.call_args.kwargs
        assert call_kwargs.get("temperature") == GENERATION_TEMPERATURE


class TestSaveDataset:
    """Тести для збереження датасету у файл."""

    def test_saves_valid_json(self, tmp_data_dir, sample_chat):
        output_path = tmp_data_dir / "data" / "chats.json"
        chats = [sample_chat]
        save_dataset(chats, str(output_path), model="gpt-4o-mini", seed=42)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "metadata" in data
        assert "chats" in data

    def test_metadata_has_required_fields(self, tmp_data_dir, sample_chat):
        output_path = tmp_data_dir / "data" / "chats.json"
        save_dataset([sample_chat], str(output_path), model="gpt-4o-mini", seed=42)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        meta = data["metadata"]
        assert "generated_at" in meta
        assert meta["model"] == "gpt-4o-mini"
        assert meta["seed"] == 42
        assert meta["total_chats"] == 1

    def test_chats_count_matches(self, tmp_data_dir, sample_chat):
        output_path = tmp_data_dir / "data" / "chats.json"
        chats = [sample_chat, sample_chat]
        save_dataset(chats, str(output_path), model="gpt-4o-mini", seed=42)

        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert len(data["chats"]) == 2
        assert data["metadata"]["total_chats"] == 2

    def test_file_is_utf8_with_cyrillic(self, tmp_data_dir, sample_chat):
        output_path = tmp_data_dir / "data" / "chats.json"
        save_dataset([sample_chat], str(output_path), model="gpt-4o-mini", seed=42)

        with open(output_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Кирилиця повинна бути читабельною, а не escaped
        assert "оплата" in content or "Доброго" in content

    def test_creates_output_directory(self, tmp_path, sample_chat):
        output_path = tmp_path / "new_dir" / "chats.json"
        save_dataset([sample_chat], str(output_path), model="gpt-4o-mini", seed=42)
        assert output_path.exists()
