import os
import streamlit as st
from langchain.chains import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
from streamlit_chat import message
from datetime import datetime

# This project is inspired by code with price youtube channel

def set_api_key():
  """Sets the OpenAI API key from Streamlit secrets."""
  os.environ['OPENAI_API_KEY'] = st.secrets["OPENAI_API_KEY"]

def get_current_datetime():
    # Get the current date and time
    now = datetime.now()
    # Format as "YYYY-MM-DD HH:MM:SS"
    datetime_string = now.strftime("%Y-%m-%d %H:%M:%S")
    return datetime_string

# Example usage:
current_datetime = get_current_datetime()

def define_prompt_template():
  """Defines the prompt template for the chat model."""
  return PromptTemplate(
      input_variables=["chat_history", "question"],
      template="""You are kind AI agent, you are currently talking to a human
               Reply in a friendly tone and like a friendly human.
               chat_history: {chat_history}
               Human: {question}
               AI:""",
  )


def initialize_chat_components():
  """Initializes the chat model, memory, and chain."""
  llm = ChatOpenAI()
  memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5)
  prompt = define_prompt_template()
  return LLMChain(llm=llm, memory=memory, prompt=prompt)


def configure_streamlit_page():
  """Configures the Streamlit page title, icon, and layout."""
  st.set_page_config(
      page_title='ChatGPT Clone',
      page_icon='',
      layout="wide"
  )


def display_title():
  """Displays the title 'ChatGPT Clone' on the Streamlit page."""
  st.title('ChatGPT Clone')
  link_text = "code with prince"
  url = "https://www.youtube.com/@CodeWithPrince"

# Display the link using markdown syntax
  #st.markdown(f"{link_text} ([{url}])")
  #st.markdown(f"<p style='color: grey;'>This is project is inspired by {link_text}([{url}]) </p>", unsafe_allow_html=True)
  st.markdown(f"<p style='color: grey;'>This is project is inspired by <a href='{url}'>{link_text}</a></p>", unsafe_allow_html=True)  

def initialize_messages():
  """Checks if messages exist in the session state and initializes if not."""
  if "messages" not in st.session_state.keys():
      st.session_state.messages = [
          {"role": "assistant", "content": "Hello! I am a chatbot. How can I help you?"}
      ]


def display_messages():
  """Displays existing messages in the chat history."""
  for message in st.session_state.messages:
      with st.chat_message(message["role"]):
          st.write(message["content"])


def get_user_input():
  """Gets user input from the Streamlit chat input box."""
  return st.chat_input()


def process_user_input(user_prompt):
  """Adds user input to messages and displays it if not None."""
  if user_prompt is not None:
      st.session_state.messages.append({
          "role": "user",
          "content": user_prompt
      })
      with st.chat_message("user"):
          st.write(user_prompt)


def get_ai_response(llm_chain, user_prompt):
  """Gets a response from the AI if the last message wasn't from the assistant."""
  if st.session_state.messages[-1]["role"] != "assistant":
      with st.chat_message("assistant"):
          with st.spinner("Loading..."):
              ai_response = llm_chain.predict(question=user_prompt)
              st.write(ai_response)
              return {"role": "assistant", "content": ai_response}
  return None


def update_messages(ai_response):
  """Appends the AI response to the messages list if not None."""
  if ai_response:
      st.session_state.messages.append(ai_response)

def main():
  """Main function to run the application."""
  set_api_key()
  llm_chain = initialize_chat_components()
  configure_streamlit_page()
  display_title()
  initialize_messages()
  display_messages()

  user_prompt = get_user_input()
  process_user_input(user_prompt)

  ai_response = get_ai_response(llm_chain, user_prompt)
  update_messages(ai_response) 

if __name__ == "__main__":
  main()