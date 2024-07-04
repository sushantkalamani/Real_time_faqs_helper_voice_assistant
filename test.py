import streamlit as st
import langchain_helper as lh

st.title("FaQ's Helper")

if "messages" not in st.session_state:
    st.session_state.messages = []

if not st.session_state.messages:
    initial_message = "Hello there! How can I assist you today?"
    st.session_state.messages.append({"role": "assistant", "content": initial_message})

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is up?"):
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    chain = lh.get_chain()
    response = chain(prompt)['result']

    with st.chat_message("assistant"):
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})

st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://i.pinimg.com/474x/49/4d/2e/494d2e25fad7412b4f11beb7242ba804.jpg");
    }
   </style>
    """,
    unsafe_allow_html=True
)