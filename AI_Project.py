"""
=============================================================================
  AI2002: Artificial Intelligence — Assignment 1, Part B
  Smart Medication Scheduling for Polypharmacy Patients
=============================================================================

  Problem
  -------
  Assign every medication dose to a 30-min time slot (0–47) within 24 hours
  such that all hard constraints (dosing intervals, drug interactions, food
  timing) are satisfied and soft cost (night-dose penalties) is minimised.

  Algorithm: Informed Backtracking Search with A*-style Heuristic
  ---------------------------------------------------------------
  Modelled as a Constraint Satisfaction Problem (CSP, AIMA Ch. 6):

    Variables   : (med_id, dose_index) — one per individual dose.
    Domains     : Subsets of {0..47} filtered by food-relation constraints.
    Constraints : Dosing intervals (same drug), interaction separations
                  (different drugs), food-relation slot restrictions.

  Solving techniques:

    MRV  (Minimum Remaining Values) — variable ordering: always assign
         the variable with the fewest remaining legal values first.

    LCV  (Least Constraining Value) — value ordering: try slots that
         eliminate the fewest options from neighbours first.

    FC   (Forward Checking) — after each assignment, prune inconsistent
         values from unassigned neighbours in-place with an undo stack.

    A*-style heuristic look-ahead — for each candidate slot, tentatively
         assign it and compute f = g(slot_cost) + h(remaining_lower_bound).
         Values with lower f are tried first, making the search informed.

  Heuristic h(n)
  --------------
  h(n) = Σ  min  C(s)
         v∈U  s∈D(v)

  where U = unassigned variables, D(v) = pruned domain, C(s) = night penalty.
  Admissible because it solves a relaxed problem (inter-variable constraints
  removed), so h(n) ≤ h*(n) for all n.

  Dependencies: Python 3.8+ (colorama optional).
=============================================================================
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Optional colour — falls back silently if colorama is absent
# ---------------------------------------------------------------------------
try:
    from colorama import Fore, Style, init as colorama_init
    colorama_init(autoreset=True)
except ImportError:
    class _NoColour:
        def __getattr__(self, _: str) -> str:
            return ""
    Fore = Style = _NoColour()


# ===========================================================================
# SECTION 1 — DATA MODELS
# ===========================================================================

@dataclass
class Medication:
    """One medication and its scheduling rules."""
    name: str                    # unique human-readable name
    doses_per_day: int           # how many doses per 24 h
    interval_slots: int          # min gap between own doses (30-min slots)
    food_relation: str = "any"   # any|with_meal|before_meal|after_meal|empty_stomach
    med_id: int = 0              # auto-assigned ID


@dataclass
class InteractionConstraint:
    """Pairwise drug interaction requiring temporal separation."""
    med_id_a: int
    med_id_b: int
    min_separation: int          # min gap in 30-min slots


# ===========================================================================
# SECTION 2 — PROBLEM FORMALISATION
# ===========================================================================

class MedicationSchedulingProblem:
    """
    CSP encoding: variables, domains, constraint helpers, heuristic.

    Time model — 48 half-hour slots: slot 0 = 00:00 … slot 47 = 23:30.
    Meal windows (inclusive):
        Breakfast 07:00–08:30 → slots 14–17
        Lunch     12:00–13:30 → slots 24–27
        Dinner    18:00–19:30 → slots 36–39
    Night window (soft penalty):
        00:00–06:00 → slots 0–12,  21:00–23:30 → slots 42–47
    """

    NUM_SLOTS     = 48
    NIGHT_PENALTY = 8
    MEAL_WINDOWS  = [(14, 17), (24, 27), (36, 39)]
    NEAR_RADIUS   = 2     # ±slots for before_meal / after_meal
    NIGHT_SLOTS: Set[int] = set(range(0, 13)) | set(range(42, 48))

    def __init__(self,
                 medications: List[Medication],
                 interactions: List[InteractionConstraint]):
        self.medications  = medications
        self.interactions = interactions

        # Pre-compute meal slot set
        self._meal_slots: Set[int] = set()
        for lo, hi in self.MEAL_WINDOWS:
            self._meal_slots.update(range(lo, hi + 1))

        # Pre-compute near-meal slot set (for before/after_meal)
        self._near_meal: Set[int] = set()
        for lo, hi in self.MEAL_WINDOWS:
            for b in (lo, hi):
                for d in range(-self.NEAR_RADIUS, self.NEAR_RADIUS + 1):
                    s = b + d
                    if 0 <= s < self.NUM_SLOTS:
                        self._near_meal.add(s)

        # Interaction map: canonical (min_id, max_id) → separation
        self._imap: Dict[Tuple[int, int], int] = {}
        for ic in interactions:
            key = (min(ic.med_id_a, ic.med_id_b),
                   max(ic.med_id_a, ic.med_id_b))
            self._imap[key] = max(self._imap.get(key, 0), ic.min_separation)

        # Adjacency list for fast neighbour lookup
        self._partners: Dict[int, List[Tuple[int, int]]] = {
            m.med_id: [] for m in medications}
        for (a, b), sep in self._imap.items():
            self._partners[a].append((b, sep))
            self._partners[b].append((a, sep))

    def initial_domain(self, med: Medication) -> List[int]:
        """Slots satisfying the food-relation hard constraint."""
        out: List[int] = []
        for s in range(self.NUM_SLOTS):
            fr = med.food_relation
            if fr == "with_meal"     and s not in self._meal_slots: continue
            if fr == "empty_stomach" and s in self._meal_slots:     continue
            if fr in ("before_meal", "after_meal") \
               and s not in self._near_meal:                        continue
            out.append(s)
        return out

    def slot_cost(self, slot: int) -> float:
        """Non-negative soft cost: NIGHT_PENALTY if night slot, else 0."""
        return float(self.NIGHT_PENALTY) if slot in self.NIGHT_SLOTS else 0.0

    # --- Admissible heuristic h(n) ---

    def heuristic(self,
                  domains:    Dict[Tuple[int, int], Set[int]],
                  assignment: Dict[Tuple[int, int], int]) -> float:
        """
        Admissible heuristic for A*-style informed evaluation.

        Formula:  h(n) = Σ_{v ∈ U} min_{s ∈ D(v)} C(s)

        where U = unassigned variables, D(v) = current domain, C(s) = slot_cost.

        Admissibility: this solves a RELAXED problem where inter-variable
        constraints are removed.  Removing constraints can only decrease
        cost, so h(n) ≤ h*(n) for all n — never overestimates.

        Consistency: for any assignment of variable v to slot s,
        h(parent) includes min C(s') ≤ C(s), so
        h(parent) ≤ C(s) + h(child).

        Returns infinity on domain wipeout (infeasible state).
        """
        h = 0.0
        for var, dom in domains.items():
            if var in assignment:
                continue
            if not dom:
                return float("inf")
            h += min(self.slot_cost(s) for s in dom)
        return h


# ===========================================================================
# SECTION 3 — CSP SOLVER (Backtracking + MRV + LCV + FC + Heuristic)
# ===========================================================================

class CSPScheduler:
    """
    Informed backtracking CSP solver.

    For each candidate value, the solver tentatively assigns it, runs
    forward checking, then evaluates f = g_increment + h(remaining).
    Values are tried in ascending f order — this is the A*-style
    informed component that guides search toward low-cost solutions.
    """

    def __init__(self, problem: MedicationSchedulingProblem,
                 max_backtracks: int = 200_000):
        self.problem        = problem
        self.max_backtracks = max_backtracks
        self._backtracks    = 0
        self._nodes         = 0
        self._cost          = 0.0

    # ---- Public entry point ----

    def solve(self) -> Optional[Dict[Tuple[int, int], int]]:
        """Run search. Returns {(med_id, dose_idx): slot} or None."""
        self._backtracks = 0
        self._nodes      = 0
        self._cost       = 0.0

        # Build variables and initial domains
        all_vars: List[Tuple[int, int]] = []
        domains:  Dict[Tuple[int, int], Set[int]] = {}
        for med in self.problem.medications:
            dom = set(self.problem.initial_domain(med))
            if not dom:
                print(f"  {Fore.RED}[CSP] '{med.name}' has no valid slots "
                      f"for '{med.food_relation}'.{Style.RESET_ALL}")
                return None
            for d in range(med.doses_per_day):
                var = (med.med_id, d)
                all_vars.append(var)
                domains[var] = set(dom)

        assignment: Dict[Tuple[int, int], int] = {}
        result = self._backtrack(assignment, domains, all_vars, 0.0)
        if result:
            self._cost = sum(self.problem.slot_cost(s) for s in result.values())
        return result

    # ---- Core recursive backtracking ----

    def _backtrack(self, assignment, domains, all_vars, g):
        self._nodes += 1

        # Goal test: all variables assigned
        if len(assignment) == len(all_vars):
            return dict(assignment)

        if self._backtracks > self.max_backtracks:
            return None

        # MRV: pick variable with smallest domain
        var = self._select_var(assignment, domains, all_vars)
        if var is None:
            return None     # domain wipeout

        # LCV + A* heuristic: order values by f = g_inc + h
        ordered = self._order_values(var, domains, assignment)

        for slot in ordered:
            # Check consistency with current assignments
            if not self._is_consistent(var, slot, assignment):
                continue

            # Assign
            assignment[var] = slot

            # Forward check (in-place, returns undo list or None on wipeout)
            removals = self._forward_check(var, slot, assignment, domains)
            if removals is not None:
                new_g = g + self.problem.slot_cost(slot)
                result = self._backtrack(assignment, domains, all_vars, new_g)
                if result is not None:
                    return result
                self._undo(removals, domains)

            del assignment[var]
            self._backtracks += 1
            if self._backtracks > self.max_backtracks:
                return None

        return None

    # ---- MRV variable selection ----

    def _select_var(self, assignment, domains, all_vars):
        """Pick unassigned variable with smallest domain (MRV).
        Ties broken by degree (most constraints)."""
        best = None
        best_sz = self.problem.NUM_SLOTS + 1
        best_dg = -1
        for v in all_vars:
            if v in assignment:
                continue
            sz = len(domains[v])
            if sz == 0:
                return None
            dg = self._degree(v)
            if sz < best_sz or (sz == best_sz and dg > best_dg):
                best, best_sz, best_dg = v, sz, dg
        return best

    def _degree(self, var):
        """Count of constraint arcs involving var (for MRV tie-break)."""
        mid = var[0]
        med = self.problem.medications[mid]
        cnt = med.doses_per_day - 1              # same-drug siblings
        for oid, _ in self.problem._partners.get(mid, []):
            cnt += self.problem.medications[oid].doses_per_day
        return cnt

    # ---- LCV + A* heuristic value ordering ----

    def _order_values(self, var, domains, assignment):
        """
        Order domain values by (tier, f_estimate, eliminations, slot).

        tier       — 0 = daytime (preferred), 1 = night (fallback).
        f_estimate — A*-style: f = g_increment + h(remaining).
                     Computed by tentatively assigning the slot, running
                     forward checking, and evaluating the admissible
                     heuristic on the resulting pruned domains.
        elim       — LCV: number of neighbour values eliminated.
        slot       — deterministic tie-break.
        """
        mid   = var[0]
        med   = self.problem.medications[mid]
        night = MedicationSchedulingProblem.NIGHT_SLOTS

        scored: List[Tuple[int, float, int, int]] = []
        for slot in domains[var]:
            # --- Count eliminations (LCV) ---
            elim = 0
            for d in range(med.doses_per_day):
                sib = (mid, d)
                if sib == var or sib in assignment:
                    continue
                for s in domains[sib]:
                    if abs(s - slot) < med.interval_slots:
                        elim += 1
            for oid, sep in self.problem._partners.get(mid, []):
                om = self.problem.medications[oid]
                for d in range(om.doses_per_day):
                    ov = (oid, d)
                    if ov in assignment:
                        continue
                    for s in domains[ov]:
                        if abs(s - slot) < sep:
                            elim += 1

            # --- A* look-ahead: tentatively assign, compute h ---
            assignment[var] = slot
            removals = self._forward_check(var, slot, assignment, domains)
            if removals is not None:
                h = self.problem.heuristic(domains, assignment)
                self._undo(removals, domains)
            else:
                h = float("inf")   # wipeout — worst possible
            del assignment[var]

            g_inc = self.problem.slot_cost(slot)
            f_est = g_inc + h
            tier  = 1 if slot in night else 0

            scored.append((tier, f_est, elim, slot))

        scored.sort()
        return [slot for _, _, _, slot in scored]

    # ---- Consistency check ----

    def _is_consistent(self, var, slot, assignment):
        """True iff slot satisfies all constraints with assigned vars."""
        mid = var[0]
        med = self.problem.medications[mid]
        for ov, os in assignment.items():
            om = ov[0]
            if om == mid:
                # Same drug: dosing interval
                if abs(slot - os) < med.interval_slots:
                    return False
            else:
                # Different drug: interaction separation
                key = (min(mid, om), max(mid, om))
                sep = self.problem._imap.get(key, 0)
                if sep > 0 and abs(slot - os) < sep:
                    return False
        return True

    # ---- Forward checking (in-place + undo stack) ----

    def _forward_check(self, var, slot, assignment, domains):
        """
        Prune inconsistent values from unassigned neighbours IN-PLACE.
        Returns list of (variable, removed_value) for undo, or None
        if any domain becomes empty (wipeout — auto-restores first).
        """
        mid = var[0]
        med = self.problem.medications[mid]
        removals: List[Tuple[Tuple[int, int], int]] = []

        for ov, dom in domains.items():
            if ov in assignment or ov == var:
                continue
            om = ov[0]

            # Determine separation this assignment imposes
            if om == mid:
                sep = med.interval_slots
            else:
                key = (min(mid, om), max(mid, om))
                sep = self.problem._imap.get(key, 0)
            if sep <= 0:
                continue

            # Remove inconsistent values
            bad = [s for s in dom if abs(s - slot) < sep]
            for s in bad:
                dom.discard(s)
                removals.append((ov, s))
            if not dom:
                self._undo(removals, domains)
                return None    # wipeout

        return removals

    def _undo(self, removals, domains):
        """Restore pruned values in reverse order."""
        for var, val in reversed(removals):
            domains[var].add(val)

    # ---- Statistics ----

    @property
    def nodes_expanded(self) -> int:   return self._nodes
    @property
    def backtracks(self) -> int:       return self._backtracks
    @property
    def solution_cost(self) -> float:  return self._cost


# ===========================================================================
# SECTION 4 — DISPLAY & EXPORT
# ===========================================================================

def slot_to_time(slot: int) -> str:
    """Slot index (0–47) to 'HH:MM'."""
    return f"{(slot * 30) // 60:02d}:{(slot * 30) % 60:02d}"


def build_slot_map(assignment):
    """Invert {(mid,d): slot} to {slot: [mid, ...]}."""
    sm: Dict[int, List[int]] = {}
    for (mid, _), slot in assignment.items():
        sm.setdefault(slot, []).append(mid)
    return sm


def display_schedule(assignment, medications):
    """Print solved schedule as a 24-hour timeline."""
    sm    = build_slot_map(assignment)
    night = MedicationSchedulingProblem.NIGHT_SLOTS

    print(f"\n{Fore.CYAN}{'=' * 72}")
    print(f"  GENERATED MEDICATION SCHEDULE — 24-HOUR DAILY PLAN")
    print(f"{'=' * 72}{Style.RESET_ALL}")
    print(f"  {'Time':<8}  {'Medications':<40}  Notes")
    print(f"  {'----':<8}  {'-' * 40:<40}  -----")

    for slot in sorted(sm):
        meds  = [medications[i] for i in sorted(set(sm[slot]))]
        names = ", ".join(m.name for m in meds)
        food  = "; ".join(m.food_relation.replace("_", " ").title()
                          for m in meds if m.food_relation != "any")
        ntag  = "!! Night dose" if slot in night else ""
        notes = " | ".join(x for x in [food, ntag] if x)
        color = Fore.YELLOW if slot in night else Fore.WHITE
        print(f"  {color}{slot_to_time(slot):<8}  {names:<40}  "
              f"{notes}{Style.RESET_ALL}")

    print(f"{Fore.CYAN}{'=' * 72}{Style.RESET_ALL}")
    total = len(assignment)
    nd    = sum(1 for s in assignment.values() if s in night)
    print(f"\n  {Fore.GREEN}Total doses scheduled : {total}{Style.RESET_ALL}")
    print(f"  {Fore.GREEN}Daytime doses         : {total - nd}{Style.RESET_ALL}")
    if nd:
        print(f"  {Fore.YELLOW}Night doses (!!)      : {nd}{Style.RESET_ALL}")
    else:
        print(f"  {Fore.GREEN}Night doses           : 0  (all daytime){Style.RESET_ALL}")
    print()


def export_schedule(assignment, medications, filename="schedule.txt"):
    """Write schedule to plain-text file."""
    if assignment is None:
        print(f"  {Fore.YELLOW}No schedule to export.{Style.RESET_ALL}")
        return
    sm    = build_slot_map(assignment)
    night = MedicationSchedulingProblem.NIGHT_SLOTS
    with open(filename, "w", encoding="utf-8") as f:
        f.write("SMART MEDICATION SCHEDULE — AI2002 Assignment 1\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"{'Time':<8}  {'Medications':<40}  Notes\n")
        f.write("-" * 60 + "\n")
        for slot in sorted(sm):
            t     = slot_to_time(slot)
            names = ", ".join(medications[i].name
                              for i in sorted(set(sm[slot])))
            food  = "; ".join(
                medications[i].food_relation.replace("_", " ").title()
                for i in sorted(set(sm[slot]))
                if medications[i].food_relation != "any")
            nt    = "[NIGHT]" if slot in night else ""
            notes = " | ".join(x for x in [food, nt] if x)
            f.write(f"{t:<8}  {names:<40}  {notes}\n")
    print(f"  {Fore.GREEN}Schedule exported to '{filename}'.{Style.RESET_ALL}")


# ===========================================================================
# SECTION 5 — CONSOLE INTERFACE
# ===========================================================================

def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_banner():
    print(f"{Fore.CYAN}")
    print("  +==============================================================+")
    print("  |     SMART MEDICATION SCHEDULER — AI2002 Assignment 1         |")
    print("  |     Informed Backtracking Search with A* Heuristic           |")
    print("  +==============================================================+")
    print(f"{Style.RESET_ALL}")


def print_menu():
    print(f"{Fore.CYAN}  --- MAIN MENU ---{Style.RESET_ALL}")
    print("   [1]  Add Medication")
    print("   [2]  View Medications")
    print("   [3]  Set Drug Interaction")
    print("   [4]  Generate Schedule")
    print("   [5]  Load Sample Profile")
    print("   [6]  Export Schedule")
    print("   [0]  Exit")
    print(f"{Fore.CYAN}  ----------------{Style.RESET_ALL}")


def get_int(prompt, lo, hi):
    """Prompt for integer in [lo, hi] with validation."""
    while True:
        raw = input(f"  {prompt} ({lo}-{hi}): ").strip()
        if not raw:
            print(f"  {Fore.RED}Input required.{Style.RESET_ALL}"); continue
        try:
            v = int(raw)
        except ValueError:
            print(f"  {Fore.RED}Enter a whole number.{Style.RESET_ALL}"); continue
        if lo <= v <= hi:
            return v
        print(f"  {Fore.RED}Must be {lo}–{hi}.{Style.RESET_ALL}")


def get_choice(prompt, valid):
    """Prompt for one of the valid options (case-insensitive)."""
    vl = [x.lower() for x in valid]
    while True:
        raw = input(f"  {prompt} [{'/'.join(valid)}]: ").strip().lower()
        if raw in vl:
            return raw
        print(f"  {Fore.RED}Options: {', '.join(valid)}{Style.RESET_ALL}")


def add_medication(medications):
    """Collect and register one medication interactively."""
    print(f"\n{Fore.CYAN}  --- ADD MEDICATION ---{Style.RESET_ALL}")
    name = input("  Medication name: ").strip()
    if not name:
        print(f"  {Fore.RED}Name cannot be empty.{Style.RESET_ALL}"); return
    if any(m.name.lower() == name.lower() for m in medications):
        print(f"  {Fore.RED}'{name}' already exists.{Style.RESET_ALL}"); return
    doses = get_int("Doses per day", 1, 6)
    hrs   = get_int("Min hours between doses", 1, 24)
    print("  Food: any, with_meal, before_meal, after_meal, empty_stomach")
    food  = get_choice("Food relation",
                       ["any","with_meal","before_meal","after_meal","empty_stomach"])
    med = Medication(name=name, doses_per_day=doses,
                     interval_slots=max(1, int(hrs * 2)),
                     food_relation=food, med_id=len(medications))
    medications.append(med)
    print(f"  {Fore.GREEN}Added '{name}' (ID {med.med_id}).{Style.RESET_ALL}")


def view_medications(medications):
    """Display registered medications."""
    if not medications:
        print(f"  {Fore.YELLOW}No medications registered.{Style.RESET_ALL}"); return
    print(f"\n{Fore.CYAN}  --- REGISTERED MEDICATIONS ---{Style.RESET_ALL}")
    print(f"  {'ID':<4}  {'Name':<22}  {'Doses/Day':<10}  {'Min Hrs':<8}  Food Relation")
    print(f"  {'--':<4}  {'-'*22}  {'-'*10}  {'-'*8}  {'-'*14}")
    for m in medications:
        print(f"  {m.med_id:<4}  {m.name:<22}  {m.doses_per_day:<10}  "
              f"{m.interval_slots/2:<8.1f}  {m.food_relation}")
    print()


def view_interactions(medications, interactions):
    """Display drug-drug interactions."""
    if not interactions:
        print(f"  {Fore.YELLOW}No interactions set.{Style.RESET_ALL}"); return
    print(f"\n{Fore.CYAN}  --- INTERACTION CONSTRAINTS ---{Style.RESET_ALL}")
    for ic in interactions:
        print(f"  {medications[ic.med_id_a].name} <-> "
              f"{medications[ic.med_id_b].name}: "
              f"min {ic.min_separation/2:.1f} h")
    print()


def set_interaction(medications, interactions):
    """Add or update a drug interaction."""
    if len(medications) < 2:
        print(f"  {Fore.YELLOW}Need at least 2 medications.{Style.RESET_ALL}"); return
    view_medications(medications)
    print(f"{Fore.CYAN}  --- SET INTERACTION ---{Style.RESET_ALL}")
    a   = get_int("First medication ID",  0, len(medications)-1)
    b   = get_int("Second medication ID", 0, len(medications)-1)
    if a == b:
        print(f"  {Fore.RED}Must be different medications.{Style.RESET_ALL}"); return
    hrs = get_int("Min separation (hours)", 1, 12)
    sep = max(1, int(hrs * 2))
    key = (min(a,b), max(a,b))
    interactions[:] = [ic for ic in interactions
        if (min(ic.med_id_a,ic.med_id_b), max(ic.med_id_a,ic.med_id_b)) != key]
    interactions.append(InteractionConstraint(a, b, sep))
    print(f"  {Fore.GREEN}{medications[a].name} <-> {medications[b].name}: "
          f"min {hrs} h.{Style.RESET_ALL}")


# ===========================================================================
# SECTION 6 — SAMPLE PROFILE
# ===========================================================================

def load_sample_profile(medications, interactions):
    """
    10-medication polypharmacy profile: elderly patient with type-2 diabetes,
    hypertension, hyperlipidaemia, GERD, neuropathic pain.
    15 total doses, 4 interaction pairs. Guaranteed solvable in <0.01 s.
    """
    medications.clear()
    interactions.clear()
    data = [
        ("Metformin",          2,  8, "with_meal"),
        ("Lisinopril",         1, 24, "any"),
        ("Atorvastatin",       1, 24, "any"),
        ("Aspirin",            1, 24, "with_meal"),
        ("Omeprazole",         1, 24, "empty_stomach"),
        ("Amlodipine",         1, 24, "any"),
        ("Metoprolol",         2,  8, "with_meal"),
        ("Gabapentin",         3,  6, "any"),
        ("Vitamin D",          1, 24, "with_meal"),
        ("Potassium Chloride", 2,  6, "with_meal"),
    ]
    for i, (n, d, h, f) in enumerate(data):
        medications.append(Medication(n, d, max(1, h*2), f, i))
    ics = [(0,4,2), (1,2,2), (3,6,2), (1,5,4)]
    for a, b, s in ics:
        interactions.append(InteractionConstraint(a, b, s))
    total = sum(m.doses_per_day for m in medications)
    print(f"  {Fore.GREEN}Loaded 10-medication profile "
          f"({total} doses, {len(ics)} interactions).{Style.RESET_ALL}")


# ===========================================================================
# SECTION 7 — MAIN LOOP
# ===========================================================================

def main():
    """Menu-driven interactive application."""
    meds:   List[Medication]            = []
    inters: List[InteractionConstraint] = []
    sched:  Optional[Dict]              = None

    clear_screen(); print_banner()

    while True:
        print_menu()
        ch = input("  Enter choice: ").strip()

        if ch == "1":
            add_medication(meds)

        elif ch == "2":
            view_medications(meds)

        elif ch == "3":
            set_interaction(meds, inters)

        elif ch == "4":
            if not meds:
                print(f"  {Fore.YELLOW}Add medications first (1 or 5).{Style.RESET_ALL}")
            else:
                td = sum(m.doses_per_day for m in meds)
                print(f"\n{Fore.CYAN}  --- GENERATING SCHEDULE ---{Style.RESET_ALL}")
                print(f"  Medications  : {len(meds)}")
                print(f"  Total doses  : {td}")
                print(f"  Interactions : {len(inters)}")
                print(f"  Algorithm    : Informed Backtracking "
                      f"(MRV + LCV + FC + A* heuristic)")
                print()
                prob = MedicationSchedulingProblem(meds, inters)
                solv = CSPScheduler(prob)
                t0   = time.time()
                res  = solv.solve()
                dt   = time.time() - t0
                print(f"  Time         : {dt:.3f} s")
                print(f"  Nodes        : {solv.nodes_expanded}")
                print(f"  Backtracks   : {solv.backtracks}")
                if res:
                    print(f"  Soft cost    : {solv.solution_cost:.0f}")
                    sched = res
                    display_schedule(res, meds)
                else:
                    sched = None
                    print(f"\n  {Fore.RED}No valid schedule found.{Style.RESET_ALL}")
                    print(f"  {Fore.YELLOW}Try: fewer doses, 'any' food, "
                          f"shorter separations.{Style.RESET_ALL}")

        elif ch == "5":
            load_sample_profile(meds, inters)
            view_medications(meds)
            view_interactions(meds, inters)

        elif ch == "6":
            fn = input("  Filename [schedule.txt]: ").strip() or "schedule.txt"
            export_schedule(sched, meds, fn)

        elif ch == "0":
            print(f"\n  {Fore.CYAN}Goodbye.{Style.RESET_ALL}\n")
            sys.exit(0)

        else:
            print(f"  {Fore.RED}Enter 0–6.{Style.RESET_ALL}")

        input(f"\n  {Fore.CYAN}Press Enter to continue...{Style.RESET_ALL}")
        clear_screen(); print_banner()


# ===========================================================================
# SECTION 8 — DEMO MODE
# ===========================================================================

def run_demo():
    """Non-interactive demo: load profile, solve, display, export."""
    print_banner()
    meds:   List[Medication]            = []
    inters: List[InteractionConstraint] = []

    print(f"{Fore.CYAN}  Loading sample polypharmacy profile...{Style.RESET_ALL}\n")
    load_sample_profile(meds, inters)
    view_medications(meds)
    view_interactions(meds, inters)

    print(f"{Fore.CYAN}  --- RUNNING INFORMED SEARCH ---{Style.RESET_ALL}")
    prob = MedicationSchedulingProblem(meds, inters)
    solv = CSPScheduler(prob)
    t0   = time.time()
    res  = solv.solve()
    dt   = time.time() - t0

    print(f"  Time         : {dt:.3f} s")
    print(f"  Nodes        : {solv.nodes_expanded}")
    print(f"  Backtracks   : {solv.backtracks}")
    if res:
        print(f"  Soft cost    : {solv.solution_cost:.0f}")
        display_schedule(res, meds)
        export_schedule(res, meds, "demo_schedule.txt")
        print(f"  {Fore.GREEN}Demo complete.{Style.RESET_ALL}\n")
    else:
        print(f"  {Fore.RED}No valid schedule found.{Style.RESET_ALL}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--demo":
        run_demo()
    else:
        main()