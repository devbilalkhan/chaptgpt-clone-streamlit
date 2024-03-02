from langchain import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.conversation.memory import (ConversationBufferMemory,
                                                  ConversationSummaryMemory,
                                                  ConversationBufferWindowMemory)
import streamlit as st
from streamlit_chat import message
import os
from datetime import datetime

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

# Initialize session state variables if they don't exist
def initialize_session_state():
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
    if 'conversation' not in st.session_state:
        st.session_state['conversation'] = None
    
# Set up the Streamlit page
def setup_page():
    st.set_page_config(page_title="ChatGPT Clone", page_icon=":speech_balloon:")
    st.markdown("<h1 style='text-align: left;'>Chat GPT Clone</h1>", unsafe_allow_html=True)
    st.sidebar.title("ChatGPT Clone:speech_balloon:")
    #st.sidebar.button("Summarize the conversation", key="summarize")

# Get a response from the model
def get_response(user_input):
    if st.session_state['conversation'] is None:
        llm = OpenAI(temperature=0, model_name='gpt-3.5-turbo-0125')
        st.session_state['conversation'] = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())
    return st.session_state['conversation'].predict(input=user_input)

# Display the conversation
def display_conversation():
    with st.container():
        for i, msg in enumerate(st.session_state['messages']):
            unique_key = f"{i}_{'user' if i % 2 == 0 else 'ai'}_{datetime.utcnow().isoformat()}"
            message(msg, is_user=i % 2 == 0, key=unique_key)

# Handle form submission
def handle_form_submission():
    with st.container():

        
        with st.form(key='my_form'):
            user_input = st.text_area("You:", height=100, max_chars=1000, key="user_input")
            if st.form_submit_button(label='Send'):
                st.session_state['messages'].append(user_input)
                model_response = get_response(user_input)
                st.session_state['messages'].append(model_response)
                st.experimental_rerun()

# Main function to run the app
def run_app():
    initialize_session_state()
    setup_page()
    display_conversation()
    handle_form_submission()
    

run_app()