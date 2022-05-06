import pyttsx3

engine = pyttsx3.init()


def read_message(text):
    engine.say(text)
    engine.runAndWait()
