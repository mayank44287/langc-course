"""
Prompt Templates and Messages in LangChain V.1
"""

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

def demo_basic_templates():
    """Basic ChatPromptTemplate usage."""

    # simple template
    simple = ChatPromptTemplate.from_template("Translate '{text}' to {language}")

    messages = simple.format_messages(text="Hello World!", language="French")
    print("Simple Template: ")
    print(f" {messages}")

    # multi-message template
    multi = ChatPromptTemplate.from_messages(
        [
            ("system", "You are translator. Be concise."),
            ("human", "translate '{text}' to {language}"),
        ]
    )

    messages = multi.format_messages(text="Good Morning", language="Japanese")
    print("\nMulti-message template:")
    for msg in messages:
        print(f" {type(msg).__name__}: {msg.content}")

def demo_prompt_composition():
    """Compose prompts from reusable parts."""

    # Reusable system prompt
    persona = ChatPromptTemplate.from_messages(
        [("system", "You are a {role}. Your tone is {tone}.")]
    )

    # Reusable task prompt
    task = ChatPromptTemplate.from_messages([("human", "{task}")])

    # Combine
    full_prompt = persona + task

    # Test different combinations
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)
    chain = full_prompt | model

    # As a pirate
    response = chain.invoke(
        {
            "role": "pirate captain",
            "tone": "adventurous",
            "task": "Tell me about your ship",
        }
    )
    print(f"Pirate: {response.content[:100]}...")

    # As a scientist
    response = chain.invoke(
        {
            "role": "scientist",
            "tone": "precise and academic",
            "task": "Explain photosynthesis",
        }
    )
    print(f"\nScientist: {response.content[:100]}...")



def demo_message_types():
    """Working with different message types."""

    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    # Build conversation manually
    messages = [
        SystemMessage(content="You are a math tutor. Be brief."),
        HumanMessage(content="What is 5*5"),
        AIMessage(content="25"),
        HumanMessage(content="And if I add 10?")
    ]

    response = model.invoke(messages)
    print(f"Conversation result: {response.content}")

def demo_messages_placeholder():
    """Use MessagesPlaceholder for dynamic conversation history."""
    prompt = ChatPromptTemplate.from_messages(
        [
        ("system", "You are a helpful assitant"),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
        ]
    )

    # simulate conversation history
    history = [
        HumanMessage(content="My name is Paulo."),
        AIMessage(content="NIce to meet you, Paulo!"),
    ]

    messages = prompt.format_messages(history=history, question= "what's my name?")

    print("With history placeholder")
    for msg in messages:
        print(f"    {type(msg).__name__}: {msg.content[:50]}...")

    # Execute
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    chain = prompt | model
    response = chain.invoke({"history": history, "question": "What's my name?"})
    
    print(f"\nResponse: {response.content}")


def demo_few_shot():
    """Few shot prompting with examples."""

    # Defining examples
    examples = [
        {"word": "happy", "opposite": "sad"},
        {"word": "fast", "opposite": "slow"},
        {"word": "big", "opposite": "small"},
    ]

    # template for each example
    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "whats th eopposite of {word} ?"),
            ("ai","The opposite of '{word}' is '{opposite}' .")
        ]
    )

    # few-shot wrapper
    few_shot = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You give the opposite of words. Follow the examples."),
            few_shot,
            ("human", "What's the opposite of '{word}'?"),
        ]
    )

    # test
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = final_prompt | model

    response = chain.invoke({"word": "bright"})
    print(f"Few-shot result: {response.content}")

if __name__ == "__main__":
    # print("=" * 50)
    # print("Demo 1: Basic Templates")
    # print("=" * 50)
    # demo_basic_templates()

    # print("\n" + "=" * 50)
    # print("Demo 2: Message Types")
    # print("=" * 50)
    # demo_message_types()

    # print("\n" + "=" * 50)
    # print("Demo 3: MessagesPlaceholder")
    # print("=" * 50)
    # demo_messages_placeholder()

    # print("\n" + "=" * 50)
    # print("Demo 4: Few-Shot")
    # print("=" * 50)
    # demo_few_shot()

    print("\n" + "=" * 50)
    print("Demo 5: Prompt Composition")
    print("=" * 50)
    demo_prompt_composition()