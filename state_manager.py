"""
NETRADRISHTI State & Queue Manager
Maintains in-memory and persistent triage queues for Rural Clinic and Ophthalmologist workstations.
"""

from datetime import datetime
from typing import Dict, List, Optional
import json
from pathlib import Path

from config import (
    PRIORITY_HIGH,
    PRIORITY_ROUTINE,
    DECISION_PENDING,
    DECISION_CONFIRM_REFER,
    DECISION_MARK_FOLLOWUP,
    DECISION_REQUEST_RECERT
)

# Persistent file path for queue store
DATA_DIR = Path(__file__).resolve().parent / "data"
QUEUE_DB_PATH = DATA_DIR / "cases_queue.json"


class StateManager:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.cases: Dict[str, dict] = {}
        self.load_from_disk()

    def add_case(
        self,
        case_id: str,
        patient_id: str,
        age: int,
        clinical_notes: str,
        image_path: str,
        quality_info: dict,
        ai_info: Optional[dict] = None
    ) -> dict:
        """Registers a case and triages it into the appropriate priority queue."""
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        priority = PRIORITY_ROUTINE
        if ai_info and ai_info.get("prediction") == "REFERABLE":
            priority = PRIORITY_HIGH

        case_record = {
            "case_id": case_id,
            "patient_id": patient_id,
            "age": age,
            "clinical_notes": clinical_notes,
            "created_at": now_str,
            "image_path": image_path,
            "quality": quality_info,
            "ai_result": ai_info,
            "priority": priority,
            "doctor_review": {
                "status": DECISION_PENDING,
                "doctor_id": None,
                "decision": None,
                "notes": "",
                "timestamp": None
            }
        }
        self.cases[case_id] = case_record
        self.save_to_disk()
        return case_record

    def update_doctor_decision(
        self,
        case_id: str,
        decision: str,
        notes: str = "",
        doctor_id: str = "Ophthalmologist Review (Tele-Medicine Desk)"
    ) -> Optional[dict]:
        """Records an ophthalmologist's final clinical decision and audit timestamp."""
        if case_id not in self.cases:
            return None

        case = self.cases[case_id]
        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        case["doctor_review"] = {
            "status": decision,
            "doctor_id": doctor_id,
            "decision": decision,
            "notes": notes,
            "timestamp": now_str
        }
        self.save_to_disk()
        return case

    def get_case(self, case_id: str) -> Optional[dict]:
        return self.cases.get(case_id)

    def get_all_cases(self) -> List[dict]:
        # Return sorted by creation timestamp descending
        return sorted(list(self.cases.values()), key=lambda c: c.get("created_at", ""), reverse=True)

    def get_high_priority_queue(self) -> List[dict]:
        return [c for c in self.get_all_cases() if c.get("priority") == PRIORITY_HIGH]

    def get_routine_queue(self) -> List[dict]:
        return [c for c in self.get_all_cases() if c.get("priority") == PRIORITY_ROUTINE]

    def save_to_disk(self):
        try:
            with open(QUEUE_DB_PATH, "w", encoding="utf-8") as f:
                json.dump(self.cases, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save cases queue to disk: {e}")

    def load_from_disk(self):
        if QUEUE_DB_PATH.exists():
            try:
                with open(QUEUE_DB_PATH, "r", encoding="utf-8") as f:
                    self.cases = json.load(f)
            except Exception:
                self.cases = {}


# Singleton instance
state_mgr = StateManager()
