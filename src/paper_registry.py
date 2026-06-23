import json
from datetime import datetime
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent.parent

REGISTRY_FILE = (
    BASE_DIR
    / "outputs"
    / "paper_registry.json"
)


def load_registry():

    if not REGISTRY_FILE.exists():
        return {}

    try:

        with open(
            REGISTRY_FILE,
            "r",
            encoding="utf-8"
        ) as file:

            return json.load(file)

    except Exception:

        return {}


def save_registry(registry):

    REGISTRY_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        REGISTRY_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            registry,
            file,
            indent=4
        )


def is_processed(
    pdf_name: str
) -> bool:

    registry = load_registry()

    return pdf_name in registry


def register_paper(
    pdf_name: str,
    pages: int,
    limitations: int = 0
):

    registry = load_registry()

    registry[pdf_name] = {
        "processed": True,
        "pages": pages,
        "limitations": limitations,
        "last_processed": datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    }

    save_registry(
        registry
    )

def get_processed_papers():

    registry = load_registry()

    return set(
        registry.keys()
    )


def update_paper(pdf_name: str, updates: dict):
    """
    Update paper entry in registry with additional fields.
    
    Args:
        pdf_name: Name of the PDF file
        updates: Dictionary of fields to update (e.g., chunk_count, embedding_count)
    """
    registry = load_registry()
    
    if pdf_name not in registry:
        registry[pdf_name] = {}
    
    # Merge updates into existing entry
    registry[pdf_name].update(updates)
    
    save_registry(registry)
