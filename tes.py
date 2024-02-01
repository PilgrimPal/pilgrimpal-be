from langchain.memory import PostgresChatMessageHistory

connection_string = f"postgresql://postgres:moshaat@127.0.0.1:6543/pilgrimpal"
memory = PostgresChatMessageHistory(
    session_id="test",
    connection_string=connection_string,
)
# memory.add_user_message("hi!")

# memory.add_ai_message("whats up?")

print(memory.messages)
