from langchain_gigachat import GigaChat
from langchain_core.messages import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv
from pathlib import Path

load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env")

def explain_prediction(llm_input):

    prompt = f"""
    Ты являешься помощником системы кредитного скоринга.

    Тебе передан результат работы ML-модели.
    Модель уже приняла решение. Твоя задача — только объяснить
    это решение понятным человеческим языком.

    ВАЖНЫЕ ПРАВИЛА:
    - Не изменяй prediction.
    - Не изменяй probability.
    - Не изменяй threshold.
    - Не придумывай данные, которых нет в переданной информации.
    - Не утверждай, что отдельный признак сам по себе гарантирует
      хороший или плохой кредитный результат.
    - SHAP показывает влияние признака на решение модели для данного клиента.
    - Положительный SHAP означает влияние в сторону Bad.
    - Отрицательный SHAP означает влияние в сторону Good.

    Объясни:
    1. Какой результат получила модель.
    2. Какова вероятность Bad.
    3. Почему модель выбрала Bad или Good с учётом threshold.
    4. Какие факторы сильнее всего повлияли на решение.
    5. Какие факторы увеличивали, а какие снижали оценку риска.

    Результат модели:

    {llm_input}

    Сформируй краткое объяснение на русском языке.
    Не используй таблицу.
    Не повторяй JSON целиком.
    """

    llm = GigaChat(
        credentials=os.getenv("GIGACHAT_CREDENTIALS"),
        scope="GIGACHAT_API_PERS",
        verify_ssl_certs=False,
        model="GigaChat-2-Max"
    )

    messages = [
        SystemMessage(content="Ты — опытный кредитный эксперт."),
        HumanMessage(content=prompt)
    ]

    response = llm.invoke(messages)
    return response.content