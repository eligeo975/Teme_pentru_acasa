from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from deepeval.metrics import GEval
from tema_3_evaluation.groq_llm import GroqDeepEval
from tema_3_evaluation.report import save_report
import sys
from dotenv import load_dotenv
import httpx
import asyncio

sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

BASE_URL = "http://127.0.0.1:8000"
THRESHOLD = 0.8

test_cases = [
    # Scenariu 1: Intrebare despre autorizatii publicitate
    LLMTestCase(
        input="Ce autorizatii sunt necesare pentru amplasarea unui panou publicitar in Romania?"
    ),
    # Scenariu 2: Intrebare despre drepturi si obligatii
    LLMTestCase(
        input="Care sunt drepturile si obligatiile proprietarului unui teren pe care se amplaseaza publicitate outdoor?"
    ),
    # Scenariu 3: Intrebare despre sanctiuni si amenzi
    LLMTestCase(
        input="Ce sanctiuni se aplica pentru publicitate stradala amplasata fara autorizatie?"
    ),
]

groq_model = GroqDeepEval()

evaluator1 = GEval(
    # Metrica 1: Relevanta raspunsului fata de legislatia romaneasca
    name="relevanta",
    criteria="""    
    Evalueaza daca raspunsul este relevant si util pentru o intrebare despre legislatia romaneasca.
    Un scor mare inseamna ca raspunsul abordeaza direct intrebarea, ofera informatii juridice corecte
    si face referire la legi, reglementari sau proceduri legale din Romania.
    Un scor mic inseamna ca raspunsul este vag, irelevant sau nu raspunde la intrebare.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)

evaluator2 = GEval(
    # Metrica 2: Lipsa bias-ului in raspuns
    name="bias",
    criteria="""    
    Evalueaza daca raspunsul este obiectiv si lipsit de bias.
    Un scor mare inseamna ca raspunsul prezinta informatiile juridice in mod neutru si echilibrat,
    fara a favoriza o anumita parte, fara opinii personale si fara a induce in eroare.
    Un scor mic inseamna ca raspunsul contine opinii subiective, favorizeaza o parte
    sau prezinta informatii intr-un mod tendentios.
    """,
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model=groq_model,
)


async def _fetch_response(client: httpx.AsyncClient, message: str, max_retries: int = 2) -> dict:
    for attempt in range(max_retries + 1):
        response = await client.post(f"{BASE_URL}/chat/", json={"message": message})
        data = response.json()
        if data.get("detail") != "Raspunsul de chat a expirat":
            return data
        if attempt < max_retries:
            await asyncio.sleep(2)
    return data


async def _run_evaluation() -> tuple[list[dict], list[float], list[float]]:
    results: list[dict] = []
    scores1: list[float] = []
    scores2: list[float] = []

    async with httpx.AsyncClient(timeout=90.0) as client:
        for i, case in enumerate(test_cases, 1):
            candidate = await _fetch_response(client, case.input)
            case.actual_output = candidate

            evaluator1.measure(case)
            evaluator2.measure(case)

            print(f"[{i}/{len(test_cases)}] {case.input[:60]}...")
            print(f"  Relevanta: {evaluator1.score:.2f} | Bias: {evaluator2.score:.2f}")

            results.append({
                "input": case.input,
                "response": candidate.get("response", str(candidate)) if isinstance(candidate, dict) else str(candidate),
                # Scorurile si motivele pentru fiecare metrica
                "relevanta_score": evaluator1.score,
                "relevanta_reason": evaluator1.reason,
                "bias_score": evaluator2.score,
                "bias_reason": evaluator2.reason,
            })
            scores1.append(evaluator1.score)
            scores2.append(evaluator2.score)

    return results, scores1, scores2


def run_evaluation() -> None:
    results, scores1, scores2 = asyncio.run(_run_evaluation())
    output_file = save_report(results, scores1, scores2, THRESHOLD)
    print(f"\nRaport salvat in: {output_file}")


if __name__ == "__main__":
    run_evaluation()
