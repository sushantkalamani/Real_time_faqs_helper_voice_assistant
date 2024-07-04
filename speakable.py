import langchain_helper as lh
import speech_recognition as sr
import pyttsx3

recognizer = sr.Recognizer()
engine = pyttsx3.init()

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

            stop_words = ["stop", "exit", "quit", "thank you"]
            querys = user_input.lower()
            if any(word in querys for word in stop_words):
                speak("Thank you for reaching!")
                print("Thank you for reaching!")
                break
            
            chain = lh.get_chain()
            response_text = chain(user_input)
            print("Bot response:", response_text['result'])

            speak(response_text['result'])

        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand audio")
        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")

    except Exception as e:
        print(f"Error: {e}")

