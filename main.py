from SpeechTranscriber import SpeechTranscriber
import asyncio
import threading
import time




def main():
    speech_transcriber = SpeechTranscriber(api_on=False)
    speech_transcriber.add_commands({
        "click": lambda: print("click command received"),
        "left click": lambda: print("click command received"),
        "right click": lambda: print("right click command received"),
        "refresh page": lambda: print("refresh page command received"),
        "screenshot": lambda: print("screenshot command received")
    })

    # async process <--------------------------------------------
    def threaded_transcribing():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(speech_transcriber.main_loop_threaded())
        finally:
            loop.close()

    def stopper(wait_for_stop=True):
        if wait_for_stop:
            listener_thread.join()  # block until the background thread is done, which can take around 1 second

    listener_thread = threading.Thread(target=threaded_transcribing)
    listener_thread.daemon = True
    listener_thread.start()
    #--------------------------------------------


    # Main thread test
    while True:
        print("Main thread is running...")
        time.sleep(1)

if __name__ == '__main__':
    asyncio.run(main())