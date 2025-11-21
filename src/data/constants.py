from typing import List, Set

CANONICAL_AA: Set[str] = set("ACDEFGHIKLMNPQRSTVWY")
INPUT_ALPHABET: List[str] = sorted(list(CANONICAL_AA.union({"X"})))
