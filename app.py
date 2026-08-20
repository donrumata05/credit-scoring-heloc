import streamlit as st
import requests


API_URL = "http://127.0.0.1:8000"


st.set_page_config(
    page_title="Кредитный скоринг",
    page_icon="🏦",
    layout="wide"
)


st.title("Кредитный скоринг")
st.markdown(
    "Введите данные клиента, чтобы получить оценку кредитного риска "
    "и объяснение решения модели."
)

FEATURE_LABELS = {
    "ExternalRiskEstimate": "Оценка внешнего кредитного риска",
    "MSinceOldestTradeOpen": "Возраст кредитной истории, мес.",
    "MSinceMostRecentTradeOpen": "Месяцев с последнего открытия счёта",
    "AverageMInFile": "Средняя длительность кредитной истории, мес.",
    "NumSatisfactoryTrades": "Количество успешных кредитных операций",
    "NumTrades60Ever2DerogPubRec": "Просрочки 60+ дней / публичные записи",
    "NumTrades90Ever2DerogPubRec": "Просрочки 90+ дней / публичные записи",
    "PercentTradesNeverDelq": "Доля операций без просрочек, %",
    "MSinceMostRecentDelq": "Месяцев с последней просрочки",
    "MaxDelq2PublicRecLast12M": "Максимальная просрочка за последние 12 мес.",
    "MaxDelqEver": "Максимальная просрочка за всё время",
    "NumTotalTrades": "Общее количество кредитных операций",
    "NumTradesOpeninLast12M": "Открыто кредитных операций за 12 мес.",
    "PercentInstallTrades": "Доля кредитов с рассрочкой, %",
    "MSinceMostRecentInqexcl7days": "Месяцев с последнего кредитного запроса",
    "NumInqLast6M": "Кредитных запросов за 6 мес.",
    "NumInqLast6Mexcl7days": "Кредитных запросов за 6 мес. без последних 7 дней",
    "NetFractionRevolvingBurden": "Доля использования возобновляемого кредита",
    "NetFractionInstallBurden": "Доля использования кредитов с рассрочкой",
    "NumRevolvingTradesWBalance": "Возобновляемых кредитов с задолженностью",
    "NumInstallTradesWBalance": "Кредитов с рассрочкой и задолженностью",
    "NumBank2NatlTradesWHighUtilization": "Кредитов с высокой загрузкой",
    "PercentTradesWBalance": "Доля кредитов с задолженностью, %"
}

with st.form("client_form"):

    st.subheader("Данные клиента")

    col1, col2 = st.columns(2)

    with col1:
        external_risk = st.number_input(
            "Оценка внешнего кредитного риска",
            min_value=0.0,
            max_value=100.0,
            value=68.0,
            help=(
                "Внешняя оценка кредитного риска клиента. "
                "Чем выше значение, тем лучше оценивается кредитный профиль."
            )
        )

        oldest_trade = st.number_input(
            "Возраст кредитной истории, мес.",
            min_value=-8.0,
            value=120.0,
            help=(
                "Количество месяцев с момента открытия самой старой "
                "кредитной операции клиента."
            )
        )

        recent_trade = st.number_input(
            "Месяцев с последнего открытия кредитной операции",
            min_value=-8.0,
            value=10.0,
            help=(
                "Количество месяцев с момента последнего открытия "
                "кредитной операции."
            )
        )

        avg_months = st.number_input(
            "Средняя длительность кредитной истории, мес.",
            min_value=-8.0,
            value=60.0,
            help=(
                "Среднее количество месяцев, в течение которых "
                "кредитные операции клиента находятся в кредитной истории."
            )
        )

        num_satisfactory = st.number_input(
            "Количество успешных кредитных операций",
            min_value=-8.0,
            value=8.0,
            help=(
                "Количество кредитных операций, которые клиент "
                "обслуживал удовлетворительно."
            )
        )

        trades_60 = st.number_input(
            "Просрочки 60+ дней / публичные записи",
            min_value=-8.0,
            value=0.0,
            help=(
                "Количество кредитных операций, связанных с просрочкой "
                "60 и более дней или соответствующими публичными записями."
            )
        )

        trades_90 = st.number_input(
            "Просрочки 90+ дней / публичные записи",
            min_value=-8.0,
            value=0.0,
            help=(
                "Количество кредитных операций, связанных с просрочкой "
                "90 и более дней или соответствующими публичными записями."
            )
        )

        percent_never_delq = st.number_input(
            "Доля операций без просрочек, %",
            min_value=-8.0,
            max_value=100.0,
            value=80.0,
            help=(
                "Процент кредитных операций клиента, по которым "
                "никогда не фиксировалась просрочка."
            )
        )

        months_since_delq = st.number_input(
            "Месяцев с последней просрочки",
            min_value=-8.0,
            value=-7.0,
            help=(
                "Количество месяцев с момента последней зарегистрированной "
                "просрочки. Значение -7 является специальным значением "
                "из исходного датасета."
            )
        )

    with col2:
        max_delq_12m = st.number_input(
            "Максимальная просрочка за последние 12 мес.",
            min_value=-8.0,
            value=0.0,
            help=(
                "Максимальный уровень просрочки или соответствующая "
                "характеристика публичных записей за последние 12 месяцев."
            )
        )

        max_delq_ever = st.number_input(
            "Максимальная просрочка за всё время",
            min_value=-8.0,
            value=0.0,
            help=(
                "Максимальный уровень просрочки, зарегистрированный "
                "за всю кредитную историю клиента."
            )
        )

        num_total_trades = st.number_input(
            "Общее количество кредитных операций",
            min_value=-8.0,
            value=10.0,
            help=(
                "Общее количество кредитных операций, присутствующих "
                "в кредитной истории клиента."
            )
        )

        trades_open_12m = st.number_input(
            "Открыто кредитных операций за последние 12 мес.",
            min_value=-8.0,
            value=2.0,
            help=(
                "Количество кредитных операций, открытых клиентом "
                "за последние 12 месяцев."
            )
        )

        percent_install = st.number_input(
            "Доля кредитов с рассрочкой, %",
            min_value=-8.0,
            max_value=100.0,
            value=50.0,
            help=(
                "Доля кредитных операций клиента, относящихся "
                "к кредитам с фиксированными платежами / рассрочкой."
            )
        )

        months_since_inq = st.number_input(
            "Месяцев с последнего кредитного запроса",
            min_value=-8.0,
            value=30.0,
            help=(
                "Количество месяцев с момента последнего запроса "
                "кредитной истории, исключая запросы за последние 7 дней."
            )
        )

        num_inq_6m = st.number_input(
            "Кредитных запросов за последние 6 мес.",
            min_value=-8.0,
            value=1.0,
            help=(
                "Количество запросов кредитной истории клиента "
                "за последние 6 месяцев."
            )
        )

        num_inq_6m_excl = st.number_input(
            "Кредитных запросов за 6 мес. без последних 7 дней",
            min_value=-8.0,
            value=1.0,
            help=(
                "Количество запросов кредитной истории за последние "
                "6 месяцев без учёта запросов, сделанных в последние 7 дней."
            )
        )

        net_revolving = st.number_input(
            "Использование возобновляемого кредита",
            min_value=-8.0,
            value=0.6,
            help=(
                "Показатель использования доступного возобновляемого "
                "кредитного лимита."
            )
        )

        net_install = st.number_input(
            "Использование кредитов с рассрочкой",
            min_value=-8.0,
            value=0.3,
            help=(
                "Показатель использования кредитов с фиксированными "
                "платежами / рассрочкой."
            )
        )

        rev_trades_balance = st.number_input(
            "Возобновляемых кредитов с задолженностью",
            min_value=-8.0,
            value=2.0,
            help=(
                "Количество возобновляемых кредитных операций, "
                "по которым имеется текущая задолженность."
            )
        )

        install_trades_balance = st.number_input(
            "Кредитов с рассрочкой и задолженностью",
            min_value=-8.0,
            value=1.0,
            help=(
                "Количество кредитных операций с фиксированными "
                "платежами, по которым имеется текущая задолженность."
            )
        )

        high_util = st.number_input(
            "Кредитов с высокой загрузкой",
            min_value=-8.0,
            value=0.0,
            help=(
                "Количество кредитных операций с высокой степенью "
                "использования доступного кредитного лимита."
            )
        )

        percent_balance = st.number_input(
            "Доля кредитов с задолженностью, %",
            min_value=-8.0,
            max_value=100.0,
            value=30.0,
            help=(
                "Процент кредитных операций, по которым имеется "
                "текущая задолженность."
            )
        )


    submitted = st.form_submit_button(
        "Рассчитать кредитный риск",
        use_container_width=True
    )

if submitted:

    client_data = {
        "ExternalRiskEstimate": external_risk,
        "MSinceOldestTradeOpen": oldest_trade,
        "MSinceMostRecentTradeOpen": recent_trade,
        "AverageMInFile": avg_months,
        "NumSatisfactoryTrades": num_satisfactory,
        "NumTrades60Ever2DerogPubRec": trades_60,
        "NumTrades90Ever2DerogPubRec": trades_90,
        "PercentTradesNeverDelq": percent_never_delq,
        "MSinceMostRecentDelq": months_since_delq,
        "MaxDelq2PublicRecLast12M": max_delq_12m,
        "MaxDelqEver": max_delq_ever,
        "NumTotalTrades": num_total_trades,
        "NumTradesOpeninLast12M": trades_open_12m,
        "PercentInstallTrades": percent_install,
        "MSinceMostRecentInqexcl7days": months_since_inq,
        "NumInqLast6M": num_inq_6m,
        "NumInqLast6Mexcl7days": num_inq_6m_excl,
        "NetFractionRevolvingBurden": net_revolving,
        "NetFractionInstallBurden": net_install,
        "NumRevolvingTradesWBalance": rev_trades_balance,
        "NumInstallTradesWBalance": install_trades_balance,
        "NumBank2NatlTradesWHighUtilization": high_util,
        "PercentTradesWBalance": percent_balance
    }


    with st.spinner("Получаем прогноз модели..."):

        try:

            response = requests.post(
                f"{API_URL}/predict",
                json=client_data,
                timeout=120
            )


            if response.status_code != 200:

                st.error(
                    f"Ошибка API: {response.text}"
                )

            else:

                result = response.json()

                st.divider()

                st.subheader("Результат оценки")


                result_col1, result_col2 = st.columns(2)

                with result_col1:

                    probability = result["probability"]
                    prediction = result["prediction"]
                    threshold = result["threshold"]


                    st.metric(
                        "Вероятность плохого кредита",
                        f"{probability:.1%}"
                    )


                    st.caption(
                        f"Порог классификации: {threshold:.1%}"
                    )


                    if prediction == "Bad":

                        st.error(
                            "Решение модели: высокий кредитный риск"
                        )

                    else:

                        st.success(
                            "Решение модели: низкий кредитный риск"
                        )

                with result_col2:

                    st.subheader("Объяснение AI")

                    st.write(
                        result["llm_explanation"]
                    )

                st.divider()

                st.subheader("Основные факторы решения")


                top_factors = result.get(
                    "top_factors",
                    []
                )


                if top_factors:

                    for factor in top_factors:

                        feature = factor["feature"]
                        value = factor["value"]
                        shap_value = factor["shap"]


                        if shap_value > 0:

                            impact = "увеличивает риск"

                        else:

                            impact = "снижает риск"


                        with st.container():

                            factor_col1, factor_col2, factor_col3 = st.columns(
                                [3, 1, 2]
                            )


                            with factor_col1:

                                st.write(
                                    f"**{feature}**"
                                )


                            with factor_col2:

                                st.write(
                                    f"`{value:g}`"
                                )


                            with factor_col3:

                                st.write(
                                    f"{impact}  \n"
                                    f"SHAP: `{shap_value:+.3f}`"
                                )


                else:

                    st.info(
                        "Информация о SHAP-факторах отсутствует."
                    )


        except requests.exceptions.ConnectionError:

            st.error(
                "Не удалось подключиться к FastAPI. "
                "Убедитесь, что API запущен на "
                f"{API_URL}"
            )


        except requests.exceptions.Timeout:

            st.error(
                "API не ответил вовремя. "
                "Генерация объяснения GigaChat может занимать некоторое время."
            )


        except Exception as e:

            st.error(
                f"Ошибка: {e}"
            )