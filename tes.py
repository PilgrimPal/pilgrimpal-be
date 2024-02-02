# from langchain.memory import PostgresChatMessageHistory

# connection_string = f"postgresql://postgres:moshaat@127.0.0.1:6543/pilgrimpal"
# memory = PostgresChatMessageHistory(
#     session_id="test",
#     connection_string=connection_string,
# )
# # memory.add_user_message("hi!")

# # memory.add_ai_message("whats up?")

# print(memory.messages)

from translator import google_translator


def translator(text: str, lang_tgt: str) -> str:
    """Translate the text to another language."""
    translator = google_translator()
    translated_text = translator.translate(text=text, lang_tgt=lang_tgt, pronounce=True)
    result = f"Translated text: {translated_text[0]}"
    if translated_text[2]:
        result += f"\nPronunciation: {translated_text[2]}"
    return result


# translator = google_translator()
# Pronounce = translator.translate(
#     "hello, where is the mosque?", lang_tgt="ar", pronounce=True
# )
# Pronounce = translator.translate(
#     "hello, where is the mosque?", lang_tgt="id", pronounce=True
# )
# print(Pronounce)

print(translator("whos ur name", "id"))