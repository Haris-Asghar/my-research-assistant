import streamlit as st
from canopy.models.data_models import UserMessage, AssistantMessage
from chatpdf import chat


def llm_chat():
    st.title("Chat with Research")

    if st.button("Clear Conversation"):
        st.session_state.messages = []

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message.role.value):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask me about your knowledge base?"):
        # st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        assistant_response, updated_history = chat(
            prompt, st.session_state.messages)
        st.session_state.messages = updated_history
        with st.chat_message("assistant"):
            st.markdown(assistant_response)


if __name__ == "__main__":
    st.set_page_config(page_title="LLM Chat", layout="wide")
    llm_chat()
