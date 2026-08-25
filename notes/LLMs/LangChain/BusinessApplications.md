# Business Applications of LangChain

## Ally Financial

Consider Ally Financial, one of the largest bank holding and car finance companies in the United States.

Ally Financial is an all-digital bank, so its employees must stay current with new technological advancements and offer innovative solutions to their customers.

To help achieve this, Ally partnered with LangChain to develop its LLM-powered platform, **Ally AI**.

This platform is designed to summarize conversations between customer support associates and customers using a large language model.

This has proven to help associates tremendously by freeing them from the repetitive task of documenting conversations, allowing them to spend more time assisting customers quickly and efficiently.

---

## Protecting Sensitive Information

Let's take a closer look at the process of summarizing conversations between customer support and Ally's customers.

Providing financial assistance inevitably involves sharing sensitive information such as:

- Customer names
- Social Security numbers
- Account details
- Email addresses
- Other personally identifiable information (PII)

Understandably, we cannot simply provide a conversation containing this information to ChatGPT and ask it to generate a summary.

Instead, we need a bridge between the conversation and the large language model that masks this personally identifiable information before passing it to the model.

This is where LangChain comes into play.

LangChain provides modules that can mask sensitive information before making the LLM call.

---

## Adyen

Another excellent use case for LangChain is Adyen.

Adyen is a fintech platform used by companies such as:

- Microsoft
- Spotify
- LinkedIn

It provides:

- End-to-end payment services
- Data-driven insights
- Financial control

Adyen has embraced AI to improve the efficiency of its customer support team.

The problem they wanted to solve was reducing response times for incoming customer support tickets through a smart ticket-routing system, or even better, automatically generating LLM-powered responses.

This results in more satisfied customers while eliminating the tedious and time-consuming process of:

- Redirecting support tickets
- Reproducing customer issues
- Constructing responses

Again, we cannot simply copy and paste a customer support ticket into ChatGPT and expect it to help the customer or determine the correct support representative.

The language model does not have access to that information.

With the help of LangChain, Adyen successfully created a support agent copilot that relies on a technique called **Retrieval-Augmented Generation (RAG)**.

---

## Robocorp

Another use case for LangChain is code completion and code generation.

Robocorp is a Python-based platform that enables users to build bots that automate tasks across many different industries.

Examples include:

- Scraping prices, reviews, and other information from websites
- Automating the monthly processing of invoices
- Automating the onboarding of new employees
- Generating financial reports
- Testing applications and websites

Building these automation bots is not an easy task.

To simplify the process, Robocorp used LangChain to build **Remark**, an AI copilot that provides coding assistance.

Remark can:

- Offer coding advice
- Generate code snippets
- Generate entire sections of code

The user simply describes the problem they want to solve using natural language, and Remark generates the appropriate code.

To make this possible, Robocorp provided the language model with data from many different sources, including coding examples and documentation.

They successfully created a product that solves customer problems quickly and efficiently.

This also reduces the workload on the support team because they now handle fewer customer support tickets and issues.

---

## Key Business Applications

We explored three interesting business applications of LangChain:

1. Masking personally identifiable information (PII)
2. Routing customer support tickets efficiently
3. Generating code using plain English