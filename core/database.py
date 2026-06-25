from __future__ import annotations
 
import json
from typing import Optional
 
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from core.config import settings


def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=settings.embed_model,
    )

def get_collection_medical(client=None):
    client = client or get_client_chromadb()
    return client.get_or_create_collection(
        name=settings.collection_medical,
        embedding_function=get_embedding_function(),
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"description": "Protocoles de traitement des plaies (RAG - Partie 5)"},
    )

def get_client_chromadb():
    return chromadb.PersistentClient(
        path=settings.chroma_path,
        settings=Settings(anonymized_telemetry=False),
    )


def _load_jsonl(jsonl_path: str) -> list[dict]:
    docs = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


def index_knowledge_base(jsonl_path: str = settings.jsonl_path, reset: bool = False) -> int:
    client = get_client_chromadb()
    if reset:
        try:
            client.delete_collection(settings.collection_medical)
        except Exception:
            pass

    col = get_collection_medical(client)
    docs = _load_jsonl(jsonl_path)

    ids = [d["id"] for d in docs]

    documents = [d["contenu"] for d in docs]
    metadatas = [
        {
            "type_plaie": d["type_plaie"],
            "categorie": d["categorie"],
            "titre": d["titre"],
            "source": d["source"],
            "mots_cles": ", ".join(d.get("mots_cles", [])),
        }
        for d in docs
    ]

    col.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return col.count()


def search_kb(query: str, type_plaie: Optional[str] = None, k: int = 3) -> list[dict]:
    col = get_collection_medical()
    where = {"type_plaie": type_plaie} if type_plaie else None
    res = col.query(query_texts=[query], n_results=k, where=where)

    resultats = []
    for i in range(len(res["ids"][0])):
        distance = res["distances"][0][i]
        resultats.append(
            {
                "id": res["ids"][0][i],
                "document": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": distance,
                "similarite": 1 - distance,
            }
        )
    return resultats


if __name__ == "__main__":
    n = index_knowledge_base(reset=True)
    print(f"[OK] {n} documents indexés dans la collection '{settings.collection_medical}'.\n")

    # 1) Recherche libre, toutes plaies confondues
    print("--- Requête libre : 'plaie qui ne cicatrise pas, faut-il consulter ?' ---")
    for r in search_kb("plaie qui ne cicatrise pas, faut-il consulter un spécialiste ?", k=3):
        m = r["metadata"]
        print(f"  [{r['similarite']:.3f}] {m['titre']} — {m['type_plaie']}/{m['categorie']}")

    # 2) Recherche filtrée par le diagnostic du CNN (ce que fera le RAG en 5.4)
    print("\n--- Diagnostic CNN = 'venous_ulcers' : protocole de traitement ---")
    for r in search_kb("traitement et prise en charge", type_plaie="venous_ulcers", k=3):
        m = r["metadata"]
        print(f"  [{r['similarite']:.3f}] {m['titre']} — {m['categorie']}")