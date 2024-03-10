import speech_recognition as sr
import os
from dotenv import load_dotenv
import sys
import asyncio
from typing import Tuple
load_dotenv()
# pip install SpeechRecognition python-dotenv openai==0.28 openai-whisper soundfile 
# pyaudio varies: https://pypi.org/project/SpeechRecognition/

class SpeechTranscriber():
    def __init__(self, api_on:bool=False, api_key:str=None):
        self.api_on = api_on
        self.api_key = api_key
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        if self.api_on and self.api_key is None:
            raise ValueError("API key must be defined when api_on is True")
        self._adjust_ambient_noise()
        # self.commands:list[str] = None
        self.commands:dict[Tuple[str], callable] = None
        self._current_command:str = None
        self.thread_callback:callable= None

    def add_thread_callback(self, functor:callable):
        self.thread_callback = functor
        
    def _adjust_ambient_noise(self):
        if not isinstance(self.recognizer, sr.Recognizer):
            raise TypeError("`recognizer` must be `Recognizer` instance")
        if not isinstance(self.microphone, sr.Microphone):
            raise TypeError("`microphone` must be `Microphone` instance")
        with self.microphone as source:
            print("Adjusting noise...")
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
            print("Done adjusting!")

    # def add_commands(self, commands:list[str]):
    #     if self.commands is None:
    #         self.commands = commands
    #     else:
    #         self.commands.extend(commands)
    def clear_commands(self):
        self.commands = None

    def add_commands(self, commands:dict[Tuple[str], callable]):
        self.commands = commands


    def listen(self, time_limit:int=30):
        audio = self.microphone_listen(time_limit)
        transcription = self.transcribe(audio)
        return transcription

    def microphone_listen(self, time_limit:int=30):
        with self.microphone as source:
            print("Say something!")
            audio = self.recognizer.listen(source=source, phrase_time_limit=time_limit)
            print("done listening!")
        return audio

    def transcribe(self, audio)->str:
        if self.api_on:
            # recognize speech using Whisper API
            try:
                transcription = self.recognizer.recognize_whisper_api(audio, api_key=self.api_key, language="english") # added language="english"
                print(f"Whisper API thinks you said: {transcription}")
            except sr.RequestError as e:
                print("Could not request results from Whisper API")
        else:
            # recognize speech using whisper
            try:
                transcription = self.recognizer.recognize_whisper(audio, language="english")
                print("Whisper thinks you said: " + transcription)
            except sr.UnknownValueError:
                print("Whisper could not understand audio")
            except sr.RequestError as e:
                print("Could not request results from Whisper")
                pass
        return transcription
    
    # async def transcribe_threaded(self, audio)->str:
    #     if self.api_on:
    #         # recognize speech using Whisper API
    #         try:
    #             transcription = self.recognizer.recognize_whisper_api(audio, api_key=self.api_key, language="english") # added language="english"
    #             print(f"Whisper API thinks you said: {transcription}")
    #         except sr.RequestError as e:
    #             print("Could not request results from Whisper API")
    #     else:
    #         # recognize speech using whisper
    #         try:
    #             transcription = self.recognizer.recognize_whisper(audio, language="english")
    #             print("Whisper thinks you said: " + transcription)
    #         except sr.UnknownValueError:
    #             print("Whisper could not understand audio")
    #         except sr.RequestError as e:
    #             print("Could not request results from Whisper")
    #             pass
    #     return transcription

    def listen_until_wake(self):
        # initializing words list
        simliar_wakeup_words = [
            "action", 
            "actions",
        ]
        while True:
            transcription = self.listen()
            keyword = None
            for greeting in simliar_wakeup_words:
                if greeting in transcription.lower():
                    print("Keyword detected...")
                    keyword = greeting
                    print("keyword")
                    break
            if keyword:
                index = transcription.lower().find(keyword)
                new_transcription = transcription[index:]
                print("wake-word-listen-substring:", transcription)
                break
        return new_transcription
    
    @property
    def current_command(self):
        return self._current_command

    @current_command.setter
    def current_command(self, value):
        self._current_command = value

    def set_current_command(self, value, transcription):
        self.current_command = value
        self.invoke_current_command(transcription)

    # def invoke_current_command(self):
    #     if self.current_command in self.commands:
    #         print('self.commands:', self.commands) # TESTING
    #         self.commands[self.current_command]()
    #     else:
    #         raise ValueError('ERROR: current command is not defined')
        
    def invoke_current_command(self, transcription):
        for command_list, command_function in self.commands.items():
            if self.current_command in command_list:
                print("transcription substr:", transcription) # TESTING
                command_function(transcription)
                break
        else:
            raise ValueError('ERROR: current command is not defined')
    
    # def listen_for_commands(self):
    #     if self.commands is None:
    #         raise ValueError("Error: initialize commands first `add_commands(array)`)")
    #     while True:
    #         transcription = self.listen()
    #         for cmd in self.commands:
    #             # add the `action` for the first word before the command
    #             if 'action ' + cmd in transcription.lower():
    #                 print(f"Command detected: {cmd}")
    #                 self.current_command = cmd
    #                 return cmd
        
    def listen_for_commands(self):
        if self.commands is None:
            raise ValueError("Error: initialize commands first `add_commands(array)`)")
        while True:
            transcription = self.listen()
            for command_list in self.commands:
                for cmd in command_list:
                    # add the `action` for the first word before the command
                    if 'action ' + cmd in transcription or 'action, ' + cmd in transcription or 'Action ' + cmd in transcription or 'Action, ' + cmd in transcription:
                        print(f"Command detected: {cmd}")
                        cmd_index = transcription.index(cmd)
                        substring = transcription[cmd_index + len(cmd):]
                        self.set_current_command(cmd, substring)
                        return self.commands[command_list]
    
    def listen_threaded(self, time_limit=30):
        if self.commands is None:
            raise ValueError("Error: initialize commands first `add_commands(array)`)")
        audio = self.microphone_listen(time_limit)
        return audio

    """
        always listens until a pause
        generate response
        continues until it deactivates somehow
    """
    def main_loop(self):
        while True:
            # listens until a pause
            command = self.listen_for_commands()
            print(command)

    async def main_loop_threaded(self):
        while True:
            audio = self.listen_threaded()
            self.thread_callback()
            # Create a task for self.transcribe(audio)
            transcribe_task = asyncio.create_task(self.transcribe(audio))
            # You can do other things here while self.transcribe(audio) is running...
            # If you need to wait for self.transcribe(audio) to finish and get its result, you can use await
            result = await transcribe_task

    

def main():
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    speech_transcriber = SpeechTranscriber(api_on=False)
    # speech_transcriber = SpeechTranscriber(api_on=True, api_key=OPENAI_API_KEY)
    # speech_transcriber.add_commands(["click", "left click", "right click", "refresh page", "screenshot"])
    speech_transcriber.add_commands({
        "click": lambda: print("click command received"),
        "left click": lambda: print("click command received"),
        "right click": lambda: print("right click command received"),
        "refresh page": lambda: print("refresh page command received"),
        "screenshot": lambda: print("screenshot command received"),
        "quit": lambda: sys.exit()
    })
    speech_transcriber.main_loop()


if __name__ == '__main__':
    main()