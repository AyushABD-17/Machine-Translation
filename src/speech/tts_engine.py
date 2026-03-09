from gtts import gTTS
import os

def speak_translation(text, save_dir="data/output"):
    """
    Transforms translated text into speech audio and saves the file.
    Returns the path to the generated audio file.
    """
    os.makedirs(save_dir, exist_ok=True)
    audio_path = os.path.join(save_dir, "translation.mp3")
    
    # We output in French since the translation target is French
    tts = gTTS(text=text, lang='fr', slow=False)
    tts.save(audio_path)
    
    return audio_path
