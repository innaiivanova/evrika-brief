"""
Quick Ragas evaluation for Evrika Briefs.

Usage (from project root):
    # 1) Install deps (once):
    #    pip install ragas datasets

    # 2) Make sure your .env has SUPABASE_URL, SUPABASE_SERVICE_KEY,
    #    OPENAI_API_KEY etc. (same as for Evrika app).

    # 3) Edit GOLD_EXAMPLES below with your own questions + ground truths.

    # 4) Run:
    #    python eval_ragas.py

This script:
- Calls your real Evrika QA endpoint: answer_question_text(question, video_hint)
- Mirrors your RAG retrieval via _match_documents (and fallback to all chunks)
- Builds a small Ragas dataset
- Runs faithfulness / answer_relevancy / context_precision / context_recall
- Prints per-sample + average scores
"""

from typing import List, Dict, Any

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import config to ensure env + clients are initialized
from evrika import config  # noqa: F401

# Import Evrika RAG pieces
from evrika.rag_pipeline import (
    answer_question_text,
    extract_youtube_id,
    _match_documents,
    _get_all_chunks_for_video,
    _is_metadata_question,
    _is_recommendation_question,
)


# -------------------------------------------------------------------
# 1) GOLD TEST SET – FILL THIS WITH YOUR REAL EXAMPLES
# -------------------------------------------------------------------

GOLD_EXAMPLES: List[Dict[str, str]] = [
    # Example structure – replace with your real questions + ground truth
    {
        "video_hint": "https://www.youtube.com/watch?v=7h5lCPpccAA",
        "question": "What is 10x thinking?",
        "ground_truth": (
            "10x thinking is a mindset focused on achieving results that are ten times greater than the current status or conventional expectations. "
            "Rather than pursuing incremental improvements (10% increases), it emphasizes radical innovation and bold, "
            "ambitious goals that require creative problem-solving and a complete reevaluation of existing assumptions and practices. "
            "This approach encourages individuals and organizations to think differently about problems, embrace risk, and foster an environment where failure is seen as a learning opportunity, "
            "ultimately aiming for significant breakthroughs in impact rather than merely optimizing current processes."
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=QwaDvbC04mI",
        "question": "What is compound interest of habits?",
        "ground_truth": (
            "Just like money earns compound interest, habits multiply over time, "
            "resulting in either positive or negative outcomes based on choices made regularly. "
            
        ),
    },
        {
        "video_hint": "https://www.youtube.com/watch?v=Mtjatz9r-Vc",
        "question": "What is jumping curves?",
        "ground_truth": (
            "Jumping Curves refers to the concept of moving beyond incremental improvements on an existing product or service (staying on the same curve) to make a significant leap in innovation that fundamentally changes the market or industry. "
            "This idea emphasizes the need to redefine and innovate by exploring completely new avenues and opportunities, rather than just making things slightly better. "
            "The speaker uses historical examples from the ice industry to illustrate how companies can fail to evolve if they cling to their original definition of their business instead of recognizing new possibilities."
            
        ),
    },
        {
        "video_hint": "https://www.youtube.com/watch?v=QFoDiAbmYXI",
        "question": "What is the build measure learn loop?",
        "ground_truth": (
            "The Build-Measure-Learn loop is a core component of the Lean Startup methodology. It involves three key steps: "
            " 1. **Build** - Create a minimum viable product (MVP) that addresses a problem or need in the market. "
            " 2. **Measure** - Collect data and feedback from customers to assess how they respond to the product. "
            " 3. **Learn** - Analyze the data to determine whether to pivot (change direction) or persevere (continue on the current path). "
            " This feedback loop aims to accelerate the learning process, allowing startups to quickly test and iterate on their ideas based on real customer input."
            
        ),
    },
        {
        "video_hint": "https://www.youtube.com/watch?v=CHxhjDPKfbY",
        "question": "What are the three core principles?",
        "ground_truth": (
            "The three core principles mentioned focus on:"
            " 1. **Getting things out of your head** - Write down everything that's on your mind to clear mental space. "
            " 2. **Defining outcomes and next actions** - Clarify what you are committed to finishing and identify the specific next steps required to move forward. "
            " 3. **Creating appropriate maps** - Establish organized structures that help visualize all projects and actions, allowing for intuitive decision-making. "
            " These principles are designed to help achieve productive engagement without the stress of a crisis. "
            
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=zGxT_sBGQCE",
        "question": "What places should I visit in Barcelona?",
        "ground_truth": (
            "In Barcelona, you should visit the following places as suggested in the video: "
            " 1. **Gothic Quarter** - Explore the old city for stunning photos and hidden corners. "
            " 2. **Sagrada Familia** - A must-see monument; you can enter the Basilica and visit the towers. "
            " 3. **Casa Mila (La Pedrera)** and **Casa Batlló** - Famous Gaudí architectural landmarks. "
            " 4. **Plaza Catalunya** - The city center and a hub for transportation. "
            " 5. **Las Ramblas** - A bustling street with shops and restaurants. "
            " 6. **La Boqueria Market** - A vibrant market for local cuisine. "
            " 7. **Barcelona Beach (Barceloneta)** - Enjoy the beach and waterfront views. "
            " 8. **Park Güell** - A beautiful park designed by Gaudí (tickets should be purchased in advance). "
            " Additionally, consider good local restaurants for tapas and meals near these attractions! "
            
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=_HQ2H_0Ayy0",
        "question": "What is a RAG?",
        "ground_truth": (
            "RAG, or Retrieval Augmented Generation, is a method that combines retrieval and generation processes to enhance the capabilities of AI assistants. It involves:"
            " 1. **Retrieval**: Converting documents and user queries into vector embeddings to enable semantic search, which matches meanings rather than keywords."
            " 2. **Augmentation**: Injecting retrieved data from a vector database into the AI's prompt at runtime, allowing it to use up-to-date information rather than static pre-trained knowledge."
            " 3. **Generation**: The AI generates responses based on the relevant data retrieved, applying its reasoning abilities to provide accurate and contextually appropriate answers."
            " This method allows for efficient querying of large datasets, improving the AI's performance in providing accurate responses. "
            
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=VVNYQKDLY5s",
        "question": "What is the advantage of a vector database?",
        "ground_truth": (
            "The advantage of a vector database is that it allows for searching based on the meaning of words rather than their exact wording. "
            "This enables more flexible and relevant search results, as users can ask questions in natural language and the database will return semantically related data. "
            "This is particularly beneficial when paired with large language models (LLMs), allowing for efficient retrieval of information without the need for the LLM to be trained specifically on the database’s structure."
            
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=rWv-KoZnpKw",
        "question": "When did the speaker work with Steve Jobs?",
        "ground_truth": (
            "The speaker worked with Steve Jobs twice in their career. The first period was from 1983 to 1987 in the Macintosh division. "
            "The second period was right after the 1997-1998 timeframe. "
            "The speaker emphasizes that their experiences with Steve Jobs were highly influential in their life and career."
            
        ),
    },
    {
        "video_hint": "https://www.youtube.com/watch?v=RUovVIU7UiA",
        "question": "What makes innovation?",
        "ground_truth": (
            "We distinguish four characteristics that make products innovative. They are: "
            " - products are deep "
            "- they are inteligent "
            "- they are complete "
            "- they are elegant "
            " Abbriviated nicely as DICE."
            
        ),
    },
    
    
    # Add 5–10 examples total for a quick evaluation
]


# -------------------------------------------------------------------
# 2) HELPER: CALL EVRIKA (ANSWER + CONTEXTS)
# -------------------------------------------------------------------

def query_evrika(question: str, video_hint: str) -> Dict[str, Any]:
    """
    Call Evrika QA and retrieve the contexts used by RAG.

    We mirror your _run_qa logic:
      - Use answer_question_text(question, video_hint) for the answer
      - Use _match_documents for retrieval
      - If no docs and youtube_id present, fall back to _get_all_chunks_for_video

    For metadata / recommendation questions, retrieval is less meaningful,
    but we still try _match_documents so Ragas can compute something.
    """
    # 1) Compute youtube_id if possible
    youtube_id = None
    if video_hint:
        try:
            youtube_id = extract_youtube_id(video_hint)
        except Exception as e:
            print(f"[EVAL] Failed to parse video_hint '{video_hint}': {e}")

    # 2) Get system answer (this uses your full _run_qa logic)
    answer = answer_question_text(question, video_hint=video_hint)

    # 3) Retrieve contexts (chunks) in a similar way to _run_qa
    docs: List[Dict[str, Any]] = []
    try:
        # Only skip retrieval if we really wanted to; but for evaluation it's
        # actually fine to always try _match_documents, so we can see retrieval quality.
        docs = _match_documents(question, youtube_id=youtube_id, match_count=6)

        if (not docs) and youtube_id:
            print(
                "[EVAL] match_documents returned 0 docs; "
                "falling back to all chunks for this video (eval)."
            )
            docs = _get_all_chunks_for_video(youtube_id)
    except Exception as e:
        print(f"[EVAL] Retrieval failed for question: {question}\n{e}")
        docs = []

    # 4) Convert docs to simple list of context strings
    contexts: List[str] = []
    if docs:
        for d in docs:
            if isinstance(d, dict):
                contexts.append(str(d.get("content", "")))
            else:
                contexts.append(str(d))
    else:
        # Fallback so Ragas still runs
        contexts = [answer]

    return {"answer": answer, "contexts": contexts}


# -------------------------------------------------------------------
# 3) BUILD RAGAS DATASET
# -------------------------------------------------------------------

def build_ragas_dataset(gold_examples: List[Dict[str, str]]) -> Dataset:
    questions: List[str] = []
    answers: List[str] = []
    gts: List[str] = []
    contexts_per_sample: List[List[str]] = []

    for ex in gold_examples:
        q = ex["question"]
        vh = ex["video_hint"]
        gt = ex["ground_truth"]

        print(f"→ Evaluating question: {q}")
        qa_result = query_evrika(q, vh)

        questions.append(q)
        answers.append(qa_result["answer"])
        gts.append(gt)
        contexts_per_sample.append(qa_result["contexts"])

    dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "ground_truth": gts,
            "contexts": contexts_per_sample,
        }
    )

    return dataset


# -------------------------------------------------------------------
# 4) RUN RAGAS
# -------------------------------------------------------------------

def run_ragas_eval(dataset: Dataset):
    """
    Run Ragas metrics on the given dataset and print results.
    """
    # LLM + embeddings used by Ragas as judge
    eval_llm = LangchainLLMWrapper(
        ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
        )
    )
    eval_emb = LangchainEmbeddingsWrapper(
        OpenAIEmbeddings(model="text-embedding-3-small")
    )

    metrics = [
        faithfulness,
        answer_relevancy,
        context_precision,
        context_recall,
    ]

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=eval_llm,
        embeddings=eval_emb,
    )

    df = result.to_pandas()
    print("\n===== RAGAS RESULTS – PER SAMPLE =====")
    print(df)

    print("\n===== RAGAS RESULTS – AVERAGES =====")
    print(df.mean(numeric_only=True))

    # Optional: save to CSV for Bootcamp slides / reporting
    try:
        df.to_csv("ragas_results.csv", index=False)
        print("\nSaved detailed results to ragas_results.csv")
    except Exception as e:
        print(f"\n[WARN] Could not save CSV: {e}")

    return df


# -------------------------------------------------------------------
# 5) MAIN
# -------------------------------------------------------------------

def main():
    if not GOLD_EXAMPLES:
        print("No GOLD_EXAMPLES defined. Please fill the GOLD_EXAMPLES list first.")
        return

    print("Building Ragas dataset from GOLD_EXAMPLES…")
    dataset = build_ragas_dataset(GOLD_EXAMPLES)

    print("\nRunning Ragas evaluation…")
    run_ragas_eval(dataset)


if __name__ == "__main__":
    main()
