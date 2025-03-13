from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate


def chat_model_basic():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    r = model.invoke("Hello, world!")
    print(r)


def chat_model_basic_conversation():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    messages = [
        SystemMessage("Solve the following math problems"),
        HumanMessage("What is 81 divided by 9?"),
        AIMessage("81 divided by 9 is 9."),
        HumanMessage("What is 10 times 5?"),
    ]
    r = model.invoke(messages)
    print(r)


def chat_model_conversation_with_user():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    chat_history = [SystemMessage("You are a helpful AI assistant.")]

    while True:
        query = input("You: ")
        chat_history.append(HumanMessage(query))

        r = model.invoke(chat_history)
        chat_history.append(AIMessage(r.content))

        print(f"AI: {r.content}")


def prompt_template_basic():
    messages = [
        ("system", "You are a comedian who tells jokes."),
        HumanMessage("Hello, Comedian! What is your name? What do you do?"),
        AIMessage(
            "I am Art the Clown. I tell creepy and scary jokes that will leave you more scared than amused. 😈"
        ),
        ("human", "Ooooh, creepy! Now tell me {joke_count} about {joke_topic}."),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"joke_count": 3, "joke_topic": "Cars"})

    print(prompt)

def prompt_template_with_chat_model():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    messages = [
        ("system", "You are a comedian who tells scary, terrifying and creepy jokes that leave the user with unease."),
        HumanMessage("Hello, Comedian! What is your name? What do you do?"),
        AIMessage(
            "I am Art the Clown. I tell creepy and scary jokes that will leave you more scared than amused."
        ),
        ("human", "Ooooh, creepy! Now tell me {joke_count} about {joke_topic}."),
    ]
    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"joke_count": 3, "joke_topic": "jesus"})

    r = model.invoke(prompt)
    print(r.content)


def main():
    print("Running main function.")
    # chat_model_basic()
    # chat_model_basic_conversation()
    # chat_model_conversation_with_user()
    # prompt_template_basic()
    prompt_template_with_chat_model()


if __name__ == "__main__":
    load_dotenv()
    main()
