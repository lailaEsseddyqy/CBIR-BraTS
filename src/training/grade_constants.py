# src/training/grade_constants.py
"""Mapping grades OMS ↔ indices pour SupCon et Classification-Guided Retrieval."""

GRADE_NAMES = ["Grade II", "Grade III", "Grade IV"]

GRADE_TO_IDX = {
    "Grade II" : 0,
    "Grade III": 1,
    "Grade IV" : 2,
}

IDX_TO_GRADE = {v: k for k, v in GRADE_TO_IDX.items()}

NUM_GRADES = len(GRADE_NAMES)
