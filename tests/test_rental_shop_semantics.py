"""Rental shop CSV semantic invariants."""

from __future__ import annotations

from source_rental_shop import OUT_DIR, verify_csv_integrity, verify_csv_semantics


def test_rental_shop_csv_semantics() -> None:
    assert OUT_DIR.is_dir(), f"missing CSV dir {OUT_DIR}"
    fk_errors = verify_csv_integrity()
    assert not fk_errors, fk_errors[:5]
    sem_errors = verify_csv_semantics()
    assert not sem_errors, sem_errors[:5]
