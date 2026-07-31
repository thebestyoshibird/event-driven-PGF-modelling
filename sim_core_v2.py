''' Creating a .py file I will import later? Why a .py file? Writes contents of the cell to a file. Why?
 So ProcessPoolExecutor can import it from worker processes across platforms (not just in Google Colab)'''

import heapq
import time
from dataclasses import dataclass, field
import numpy as np

MONTH_LEN = 30.44      # days/month (~365/12)
N_MONTHS = 36           # 36-month observation window
DEFAULT_T = 365 * 3     # 1,095 days = 3 years


class Person:
    __slots__ = (
        "amIhomeless", "num_eps", "num_months", "num_undetected_months",
        "consecutive", "episodic", "months_homeless", "months_detected",
        "this_episode_months", "ep_and_cons", "p_exit", "ever_episodic",
        "ever_consecutive", "ever_homeless", "index",
    )
    # __slots__ is a small memory/speed optimization for large populations

    def __init__(self):
        self.amIhomeless = False
        self.num_eps = 0
        self.num_months = 0
        self.num_undetected_months = 0
        self.consecutive = False
        self.episodic = False
        self.months_homeless = [False] * N_MONTHS
        self.months_detected = [True] * N_MONTHS
        self.this_episode_months = 0
        self.ep_and_cons = False
        self.p_exit = 0.0
        self.ever_episodic = False
        self.ever_consecutive = False
        self.ever_homeless = False
        self.index = 0

    def _month_index(self, t):
        return min(int(t // MONTH_LEN), N_MONTHS - 1)    # // is a floor division
                                                    # operator - returns the
                                                    # rounded-down integer from
                                                    # division

    def shouldIaddmonth(self, t, p_e_base):
        this_month = self._month_index(t)
        if not self.months_homeless[this_month]:
            for month in range(this_month, N_MONTHS):
                self.months_homeless[month] = True
            self.num_months = sum(self.months_homeless[: this_month + 1])
            # Counts total # of months homeless to determine whether > 12 in 36
            # months
            self.num_eps += 1
            self.amIhomeless = True
            self.this_episode_months += 1
            self.p_exit = p_e_base ** self.this_episode_months
            # Fixed this from original b/c v1 used number of lifetime homeless
            # months as exponent, rather than current episode only

    def amIchronic(self, t, fix_known_bugs=True):
        if self.num_months > 12 and self.num_eps >= 4:
            self.episodic = True
            self.ever_episodic = True
        if self.num_months > 12 and self.num_eps < 4:
            self.episodic = False
            this_month = self._month_index(t)
            if consecutive_test(self.months_homeless, this_month, fix_known_bugs=fix_known_bugs):
                self.consecutive = True
                self.ever_consecutive = True
            else:
                self.consecutive = False
        if self.consecutive and self.episodic:
            self.ep_and_cons = True

    def addundetectedmonth(self, t):
        this_month = self._month_index(t)
        if self.months_detected[this_month]:
            self.months_detected[this_month] = False
            self.num_undetected_months += 1

    def makemehomeless(self, t, p_undetected, p_e_base, rng, fix_known_bugs=True):    # I can include rng as an argument?!
        self.shouldIaddmonth(t, p_e_base)
        self.amIhomeless = True
        self.ever_homeless = True
        self.amIchronic(t, fix_known_bugs=fix_known_bugs)
        if rng.random() < (self.p_exit / (p_undetected + self.p_exit)):
            return "exit"
        return "undetected"

    def exit_homelessness(self, t, p_e_base, fix_known_bugs=True):
        self.amIhomeless = False
        this_month = self._month_index(t)
        for month in range(this_month + 1, N_MONTHS):
            self.months_homeless[month] = False
        self.amIchronic(t, fix_known_bugs=fix_known_bugs)
        self.p_exit = p_e_base
        self.this_episode_months = 0


def consecutive_test(months_homeless, this_month, fix_known_bugs=False):
    """Checks the trailing 13 months (inclusive) for >12 True values."""
    counter = 0
    if fix_known_bugs:
        lo = max(this_month - 13, -1)  # Yoshi: Check this
    else:
        lo = this_month - 14  # Yoshi: Check this
    for x in range(this_month, lo, -1):
        if months_homeless[x]:
            counter += 1
        if counter > 12:
            return True
    return False


def _draw_tau(rng, p_h, p_u, p_e, event_type):    # DEFINITELY CHECK THIS!!!
    if event_type == "exit":
        return rng.exponential(1.0 / p_e)
    elif event_type == "undetected":
        return rng.exponential(1.0 / p_u)
    else:
        return rng.exponential(1.0 / p_h)


@dataclass                # Creates a class primarily used to "hold state"
class RunResult:
    p_h: float
    p_e: float
    p_u: float
    sim: int
    n_consecutive: int
    n_episodic: int
    n_chronic: int
    n_ever_homeless: int
    n_undetected_chronic: int
    frac_undetected_chronic: float
    wall_time_sec: float
    times: list = field(default_factory=list)
    now_homeless_ts: list = field(default_factory=list)
    ever_homeless_ts: list = field(default_factory=list)
    consecutive_ts: list = field(default_factory=list)
    episodic_ts: list = field(default_factory=list)


def _find_undetected_chronic_vectorized(population):
    episodic_people = [p for p in population if p.ever_episodic]
    if not episodic_people:
        return 0, float("nan")

    H = np.array([p.months_homeless for p in episodic_people], dtype=bool)
    D = np.array([p.months_detected for p in episodic_people], dtype=bool)
    detected_homeless = H & D            # detected_homeless is an array where each entry is True iff H and D are both True

    detected_num_months = detected_homeless.sum(axis=1)          # Counts the number of detected homeless months (Trues) in detected_homeless
    prev = np.zeros((detected_homeless.shape[0], 1), dtype=bool)   # Creates an array of zeroes but why is dtype bool?
    shifted = np.concatenate([prev, detected_homeless[:, :-1]], axis=1)     # What is going on here?
    new_episode_starts = detected_homeless & ~shifted
    detected_num_eps = new_episode_starts.sum(axis=1)

    would_be_missed = (detected_num_eps < 4) | (detected_num_months < 12)     # "|" with numpy performs an element-wise logical OR. So if either condition is True, would_be_missed is also True.
    n_undetected = int(would_be_missed.sum())
    frac_undetected = n_undetected / len(episodic_people)
    return n_undetected, frac_undetected


def simulate_single_run(p_h, p_e_base, p_u, size_pop=200_000, T=DEFAULT_T, sim_id=0,
                          seed=None, record_timeseries=True, fix_known_bugs=False):
    """
    V2 optimized single-simulation runner. Same event-driven algorithm, RNG draw
    order, and per-agent stochastic rules as the original; changes are (a) O(1)
    running counters instead of O(N) rescans every event, and (b) the opt-in
    bug fixes described in Section 2.
    """
    _start = time.perf_counter()
    rng = np.random.default_rng(seed)

    population = [Person() for _ in range(size_pop)]
    for i, person in enumerate(population):
        person.p_exit = p_e_base
        person.index = i
        if rng.random() < p_h:
            person.amIhomeless = True
            person.ever_homeless = True
            person.shouldIaddmonth(0, p_e_base)

    now_homeless_count = sum(1 for p in population if p.amIhomeless)
    ever_homeless_count = sum(1 for p in population if p.ever_homeless)
    ever_consecutive_count = 0
    ever_episodic_count = 0

    event_queue = []
    for i, person in enumerate(population):
        if person.amIhomeless and rng.random() < p_e_base / (p_u + p_e_base):
            event_type = "exit"
        elif person.amIhomeless:
            event_type = "undetected"
        else:
            event_type = "homeless"
        tau = _draw_tau(rng, p_h, p_u, p_e_base, event_type)
        heapq.heappush(event_queue, (tau, i, event_type))

    times, now_ts, ever_ts, cons_ts, epis_ts = [], [], [], [], []
    homeless_event = "homeless"  # mirrors the ORIGINAL's single reused scalar (issue #3)

    while event_queue:
        t, idx, event = heapq.heappop(event_queue)
        if t >= T:
            break
        person = population[idx]

        if event == "homeless":
            if not person.amIhomeless:
                was_ever_h = person.ever_homeless
                was_cons, was_epis = person.ever_consecutive, person.ever_episodic
                homeless_event = person.makemehomeless(t, p_u, p_e_base, rng, fix_known_bugs=fix_known_bugs)
                now_homeless_count += 1
                if person.ever_homeless and not was_ever_h:
                    ever_homeless_count += 1
                if person.ever_consecutive and not was_cons:
                    ever_consecutive_count += 1
                if person.ever_episodic and not was_epis:
                    ever_episodic_count += 1
            else:
                person.shouldIaddmonth(t, p_e_base)
                if fix_known_bugs:
                    homeless_event = "exit" if rng.random() < (person.p_exit / (p_u + person.p_exit)) else "undetected"
                # else: intentionally NOT reassigned - reproduces issue #3.
        elif event == "undetected" and person.amIhomeless:
            person.addundetectedmonth(t)
            homeless_event = "exit" if rng.random() < (person.p_exit / (p_u + person.p_exit)) else "undetected"
        elif event == "exit" and person.amIhomeless:
            was_cons, was_epis = person.ever_consecutive, person.ever_episodic
            person.exit_homelessness(t, p_e_base, fix_known_bugs=fix_known_bugs)
            now_homeless_count -= 1
            if person.ever_consecutive and not was_cons:
                ever_consecutive_count += 1
            if person.ever_episodic and not was_epis:
                ever_episodic_count += 1
            homeless_event = "homeless"

        person.num_months = sum(person.months_homeless[: person._month_index(t) + 1])

        tau = _draw_tau(rng, p_h, p_u, person.p_exit, homeless_event)
        heapq.heappush(event_queue, (t + tau, idx, homeless_event))

        if record_timeseries:
            times.append(t)
            now_ts.append(now_homeless_count)
            ever_ts.append(ever_homeless_count)
            cons_ts.append(ever_consecutive_count)
            epis_ts.append(ever_episodic_count)

    last_month = min(int((T - 1) // MONTH_LEN), N_MONTHS - 1)
    for person in population:
        person.num_months = sum(person.months_homeless[: last_month + 1])
        person.amIchronic(T, fix_known_bugs=fix_known_bugs)

    n_consecutive = sum(1 for p in population if p.ever_consecutive)
    n_episodic = sum(1 for p in population if p.ever_episodic)
    n_ever_homeless = sum(1 for p in population if p.ever_homeless)
    n_chronic = n_consecutive + n_episodic
    n_undetected, frac_undetected = _find_undetected_chronic_vectorized(population)

    wall = time.perf_counter() - _start
    return RunResult(
        p_h=p_h, p_e=p_e_base, p_u=p_u, sim=sim_id,
        n_consecutive=n_consecutive, n_episodic=n_episodic, n_chronic=n_chronic,
        n_ever_homeless=n_ever_homeless,
        n_undetected_chronic=n_undetected, frac_undetected_chronic=frac_undetected,
        wall_time_sec=wall,
        times=times, now_homeless_ts=now_ts, ever_homeless_ts=ever_ts,
        consecutive_ts=cons_ts, episodic_ts=epis_ts,
    )
