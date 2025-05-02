import asyncio
import csv
from langchain_groq import ChatGroq
from pydantic import BaseModel
from src.pipeline import QAPipeline
from src.settings import settings


class LLMResponse(BaseModel):
    is_correct: bool
    is_idk: bool


pipeline = QAPipeline()

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    max_tokens=512,
    max_retries=2,
    timeout=30000,
    api_key=settings.GROQ_API_KEY,
)

# Bind structured evaluation output
llm_evaluator = llm.with_structured_output(LLMResponse, include_raw=True)

# Sample queries and ground truths
queries = [
    "What are the ingredients of tomato soup?",
    "Recipe for an apple pie",
    "How do you make a chocolate cake?",
    "How do you make dal bhat?",
    "What are the ingredients for making dhido?",
    "How do you make chana masala?",
    "Recipe for butter chicken",
    "Recipe for ramen",
]

ground_truths = [
    "Tomato Soup: tomatoes, onion, garlic, vegetable broth, cream, salt, pepper",
    "Apple Pie: apples, flour, butter, sugar, cinnamon",
    "I don't know",
    "I don't know",
    "I don't know",
    "Chana Masala: chickpeas, tomato, onion, garlic, ginger, spices",
    "Butter Chicken: chicken, tomato puree, cream, butter, spices, garlic, ginger",
    "Ramen: ramen noodles, broth, boiled egg, green onion, soy sauce",
]

# Evaluation function
async def evaluate_pipeline():
    correct = 0
    idk = 0
    total = len(queries)
    results = []

    for q, gt in zip(queries, ground_truths):
        response = await pipeline.answer_query_(q)
        answer = response.answer

        # Evaluation prompt
        eval_prompt = f"""
You are an evaluator. Assess whether the model's answer is both factually correct and acknowledges lack of knowledge when necessary.

Question: {q}
Model's Answer: {answer}
Ground Truth: {gt}

Evaluate the following:
1. Is the model's answer semantically correct when compared to the ground truth?
2. Does the model appropriately say "I don't know" or avoid answering if the answer is not available?

Respond in JSON with two fields:
- is_correct: true or false
- is_idk: true or false
"""

        result = llm_evaluator.invoke(eval_prompt)
        parsed = result["parsed"]

        # Correct = either factually correct or correctly says "I don't know" when GT also says so
        if parsed.is_correct or (parsed.is_idk and gt.strip().lower() == "i don't know"):
            correct += 1
        if parsed.is_idk:
            idk += 1

        # Log and store results
        print(
            f"Q: {q}\nA: {answer}\nGT: {gt}\nCorrect: {parsed.is_correct}, IDK: {parsed.is_idk}\n{'-' * 60}"
        )

        results.append(
            {
                "question": q,
                "model_answer": answer,
                "ground_truth": gt,
                "is_correct": parsed.is_correct,
                "is_idk": parsed.is_idk,
            }
        )

    # Save results to CSV
    with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=[
                "question",
                "model_answer",
                "ground_truth",
                "is_correct",
                "is_idk",
            ],
        )
        writer.writeheader()
        writer.writerows(results)

    # Print summary
    print(f"\nEvaluation results saved to 'evaluation_results.csv'.")
    print(f"Total Correct: {correct}/{total} ({(correct / total) * 100:.2f}%)")
    print(f"'I don't know' Responses: {idk}/{3} ({(idk / 3) * 100:.2f}%)") #here 3 because there a re 3 total i dont know response 

# Entry point
if __name__ == "__main__":
    asyncio.run(evaluate_pipeline())
