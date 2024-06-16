import os
from dotenv import load_dotenv
import speech_recognition as sr
import pyttsx3
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import ConversationChain

recognizer = sr.Recognizer()
engine = pyttsx3.init()

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
llm = ChatGoogleGenerativeAI(model="gemini-pro",
                             temperature=0.3)
conversation = ConversationChain(llm=llm)

def speak(text):
    engine.say(text)
    engine.runAndWait()

while True:
    try:
        with sr.Microphone() as source:
            print("Listening...")
            audio = recognizer.listen(source)

        try:
            user_input = recognizer.recognize_google(audio)
            print("You said:", user_input)

            stop_words = ["stop", "exit", "quit"]
            querys = user_input.lower()
            if any(word in querys for word in stop_words):
                print("Thank you for reaching!")
                break

            response_text = conversation.run(user_input)
            print("Bot response:", response_text)

            speak(response_text)

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    except Exception as e:
        print(f"Error: {e}")
