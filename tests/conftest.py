"""Спільні фікстури для тестів."""

import json

import pytest


@pytest.fixture
def sample_message_client():
    return {"role": "client", "text": "Доброго дня! У мене не проходить оплата картою."}


@pytest.fixture
def sample_message_agent():
    return {"role": "agent", "text": "Вітаю! Давайте розберемось з вашою ситуацією."}


@pytest.fixture
def sample_scenario_successful():
    return {
        "category": "payment_issue",
        "case_type": "successful",
        "has_hidden_dissatisfaction": False,
        "intended_agent_mistakes": [],
    }


@pytest.fixture
def sample_scenario_with_mistakes():
    return {
        "category": "refund_request",
        "case_type": "agent_error",
        "has_hidden_dissatisfaction": True,
        "intended_agent_mistakes": ["ignored_question", "no_resolution"],
    }


@pytest.fixture
def sample_chat(sample_scenario_successful):
    return {
        "id": "chat_001",
        "scenario": sample_scenario_successful,
        "messages": [
            {"role": "client", "text": "Доброго дня! У мене не проходить оплата картою."},
            {"role": "agent", "text": "Вітаю! Давайте перевіримо статус вашого платежу."},
            {"role": "client", "text": "Я пробував тричі, кожного разу помилка."},
            {"role": "agent", "text": "Перевірив — блокування зняте, спробуйте ще раз."},
            {"role": "client", "text": "Працює! Дякую за допомогу!"},
            {"role": "agent", "text": "Радий допомогти! Гарного дня!"},
        ],
    }


@pytest.fixture
def sample_chat_hidden_dissatisfaction():
    return {
        "id": "chat_042",
        "scenario": {
            "category": "technical_error",
            "case_type": "problematic",
            "has_hidden_dissatisfaction": True,
            "intended_agent_mistakes": ["no_resolution"],
        },
        "messages": [
            {"role": "client", "text": "Добрий день, у мене не працює експорт даних."},
            {"role": "agent", "text": "Вітаю! Спробуйте очистити кеш браузера."},
            {"role": "client", "text": "Вже пробував, не допомагає."},
            {"role": "agent", "text": "Тоді рекомендую переглянути наш FAQ розділ."},
            {"role": "client", "text": "Добре, дякую за інформацію. Спробую розібратись сам."},
            {"role": "agent", "text": "Звертайтесь, якщо будуть ще питання!"},
        ],
    }


@pytest.fixture
def sample_analysis_result():
    return {
        "chat_id": "chat_001",
        "intent": "payment_issue",
        "satisfaction": "satisfied",
        "quality_score": 5,
        "agent_mistakes": [],
        "summary": "Клієнт звернувся з проблемою оплати. Агент швидко вирішив питання.",
    }


@pytest.fixture
def sample_analysis_result_negative():
    return {
        "chat_id": "chat_042",
        "intent": "technical_error",
        "satisfaction": "unsatisfied",
        "quality_score": 2,
        "agent_mistakes": ["no_resolution", "ignored_question"],
        "summary": "Клієнт мав проблему з експортом. Агент не надав реального рішення.",
    }


@pytest.fixture
def sample_chats_dataset(sample_chat, sample_chat_hidden_dissatisfaction):
    return {
        "metadata": {
            "generated_at": "2026-02-23T12:00:00Z",
            "model": "gpt-4o-mini",
            "total_chats": 2,
            "seed": 42,
        },
        "chats": [sample_chat, sample_chat_hidden_dissatisfaction],
    }


@pytest.fixture
def sample_analysis_dataset(sample_analysis_result, sample_analysis_result_negative):
    return {
        "metadata": {
            "analyzed_at": "2026-02-23T12:30:00Z",
            "model": "gpt-4o",
            "total_analyzed": 2,
        },
        "results": [sample_analysis_result, sample_analysis_result_negative],
    }


@pytest.fixture
def mock_openai_chat_response():
    """Мок відповіді OpenAI API для генерації діалогу."""

    def _make_response(content: dict):
        class Choice:
            class Message:
                def __init__(self, c):
                    self.content = json.dumps(c, ensure_ascii=False)
            def __init__(self, c):
                self.message = self.Message(c)

        class Usage:
            prompt_tokens = 500
            completion_tokens = 300
            total_tokens = 800

        class Response:
            choices = [Choice(content)]
            usage = Usage()

        return Response()

    return _make_response


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Тимчасова директорія для тестових даних."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    return tmp_path
