from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnableParallel, RunnableBranch


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
        (
            "system",
            "You are a comedian who tells scary, terrifying and creepy jokes that leave the user with unease.",
        ),
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


def chains_basics():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian who tells jokes about {joke_topic}."),
            ("human", "Tell me {joke_count} jokes."),
        ]
    )

    chain = prompt_template | model | StrOutputParser()

    r = chain.invoke({"joke_topic": "lawyers", "joke_count": 3})
    print(r)


def chains_extended():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")
    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a comedian who tells jokes about {joke_topic}."),
            ("human", "Tell me {joke_count} jokes."),
        ]
    )

    uppercase_output = RunnableLambda(lambda x: x.upper())
    count_words = RunnableLambda(lambda x: f"Word count: {len(x.split())}\n{x}")

    chain = prompt_template | model | StrOutputParser() | uppercase_output | count_words

    r = chain.invoke({"joke_topic": "lawyers", "joke_count": 3})
    print(r)


def chains_parallel():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    prompt_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert product reviewer."),
            ("human", "List the main features of the product {product_name}."),
        ]
    )

    def analyze_pros(features: str):
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert product reviewer."),
                (
                    "human",
                    "Given these features: {features}, list the pros these features.",
                ),
            ]
        ).format_prompt(features=features)

    def analyze_cons(features: str):
        return ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert product reviewer."),
                (
                    "human",
                    "Given these features: {features}, list the cons these features.",
                ),
            ]
        ).format_prompt(features=features)

    pros_branch = RunnableLambda(lambda x: analyze_pros(x)) | model | StrOutputParser()
    cons_branch = RunnableLambda(lambda x: analyze_cons(x)) | model | StrOutputParser()

    chain = (
        prompt_template
        | model
        | StrOutputParser()
        | RunnableParallel(branches={"pros": pros_branch, "cons": cons_branch})
        | RunnableLambda(
            lambda x: f"Pros:\n{x['branches']['pros']}\n\nCons:\n{x['branches']['cons']}"
        )
    )

    r = chain.invoke({"product_name": "MacBook Pro"})
    print(r)


def chains_branching():
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro")

    positive_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a thank you note for this positive feedback: {feedback}.",
            ),
        ]
    )
    negative_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a response addressing this negative feedback: {feedback}.",
            ),
        ]
    )
    neutral_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a request for more details for this neutral feedback: {feedback}.",
            ),
        ]
    )
    escalate_feedback_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Generate a message to escalate this feedback to a human agent: {feedback}.",
            ),
        ]
    )

    branches = RunnableBranch(
        (
            lambda x: "positive" in x,
            positive_feedback_template | model | StrOutputParser(),
        ),
        (
            lambda x: "negative" in x,
            negative_feedback_template | model | StrOutputParser(),
        ),
        (
            lambda x: "neutral" in x,
            neutral_feedback_template | model | StrOutputParser(),
        ),
        escalate_feedback_template | model | StrOutputParser(),
    )

    classification_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            (
                "human",
                "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}.",
            ),
        ]
    )

    classification_chain = classification_template | model |  StrOutputParser()

    chain = classification_chain | branches

    # Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
    # Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
    # Neutral review - "The product is okay. It works as expected but nothing exceptional."
    # Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

    review = "The product is terrible. It broke after just one use and the quality is very poor."
    r = chain.invoke({"feedback": review})
    print(r)


def main():
    print("Running main function.")
    # chat_model_basic()
    # chat_model_basic_conversation()
    # chat_model_conversation_with_user()
    # prompt_template_basic()
    # prompt_template_with_chat_model()
    # chains_basics()
    # chains_extended()
    # chains_parallel()
    # chains_branching()

    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector = embeddings.embed_query("I love cats more than dogs.")
    print(f"Vector α has {len(vector)} dimensions.")

    vector = embeddings.embed_query("Harry porter sucks!")
    print(f"Vector β has {len(vector)} dimensions.")


if __name__ == "__main__":
    load_dotenv()
    main()
