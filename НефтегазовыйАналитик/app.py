import streamlit as st

from oil_analyst.factory import build_agent


@st.cache_resource
def get_agent():
    return build_agent()

st.set_page_config(page_title="Нефтегазовый аналитик", page_icon="🛢️", layout="wide")
st.title("Нефтегазовый аналитик")
st.caption("Отчёты имеют приоритет над интернетом; неподтверждённые цифры не генерируются.")

if "messages" not in st.session_state: st.session_state.messages = []
for message in st.session_state.messages:
    with st.chat_message(message["role"]): st.markdown(message["content"])

query = st.chat_input("Спросите о Brent, WTI, Urals, OPEC+ или прогнозе")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"): st.markdown(query)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Анализирую источники..."):
                response = get_agent().invoke(query)
            st.markdown(response.answer)
            if response.sources:
                with st.expander("Источники", expanded=True):
                    for source in response.sources: st.markdown(f"- {source.citation}")
            if response.forecast:
                st.subheader("Прогноз")
                st.line_chart({p.period: p.value for p in response.forecast.forecast})
                with st.expander("Технические данные прогноза", expanded=False):
                    st.json(response.forecast.model_dump(mode="json"))
            if response.warnings: st.warning("\n".join(response.warnings))
            if response.uncertainty: st.info(response.uncertainty)
            if response.metadata_filters.active():
                st.caption("Фильтры: " + str(response.metadata_filters.model_dump(exclude_none=True)))
            st.caption("Маршрут: " + " → ".join(response.route))
            st.session_state.messages.append({"role": "assistant", "content": response.answer})
        except Exception as exc:
            st.error(f"Не удалось обработать запрос: {exc}")
