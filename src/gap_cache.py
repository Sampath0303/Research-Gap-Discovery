import hashlib
import json
from pathlib import Path


CACHE_FILE = (
    Path(__file__).resolve().parent.parent
    / "outputs"
    / "gap_cache.json"
)


def get_cluster_hash(cluster_data):
    """Generate deterministic hash for cluster data."""
    cluster_str = json.dumps(cluster_data, sort_keys=True)
    return hashlib.md5(cluster_str.encode()).hexdigest()


def load_cache():
    if not CACHE_FILE.exists():
        return {}

    try:
        with open(
            CACHE_FILE,
            "r",
            encoding="utf-8"
        ) as f:
            return json.load(f)
    except Exception:
        return {}


def save_cache(cache):
    CACHE_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with open(
        CACHE_FILE,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            cache,
            f,
            indent=4
        )


def get_cached_gap(cluster_data):
    """Retrieve cached gap result for a cluster."""
    cache = load_cache()
    cluster_hash = get_cluster_hash(cluster_data)
    
    if cluster_hash in cache:
        print("Using cached gap")
        return cache[cluster_hash]
    
    return None


def cache_gap_result(cluster_data, gap_result):
    """Cache a generated gap result."""
    cache = load_cache()
    cluster_hash = get_cluster_hash(cluster_data)
    cache[cluster_hash] = gap_result
    save_cache(cache)