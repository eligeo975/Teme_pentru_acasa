import requests
import httpx
import sys
import pytest

# foloseste UTF-8 pentru stdout ca sa evite erori de codare
sys.stdout.reconfigure(encoding="utf-8")

BASE_URL = "http://localhost:8000"


# Test pentru endpoint-ul root
def test_root_endpoint():
    """Verifica ca endpoint-ul root returneaza mesajul asteptat."""
    response = requests.get(f"{BASE_URL}/")
    assert response.status_code == 200
    data = response.json()
    assert "message" in data
    assert "RAG Assistant ruleaza" in data["message"]


# Scenariu de testare pozitiv pentru endpoint-ul /chat/ - evaluat de LLM as a Judge
def test_chat_relevant_question():
    """Trimite o intrebare relevanta despre legislatie si verifica raspunsul."""
    response = requests.post(
        f"{BASE_URL}/chat/",
        json={"message": "Ce autorizatii sunt necesare pentru amplasarea unui panou publicitar in Romania?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Raspunsul trebuie sa contina informatii despre legislatie/autorizatii
    assert len(str(data["response"])) > 20


# Test negativ pentru endpoint-ul /chat/ - evaluat de LLM as a Judge
def test_chat_irrelevant_question():
    """Trimite o intrebare irelevanta si verifica ca asistentul o respinge."""
    response = requests.post(
        f"{BASE_URL}/chat/",
        json={"message": "Care e reteta de sarmale?"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Raspunsul trebuie sa indice ca intrebarea nu e relevanta
    resp_text = str(data["response"]).lower()
    assert "legislat" in resp_text or "juridic" in resp_text or "nu pare" in resp_text
