''' Creating a .py file I will import later? Why a .py file? Writes contents of the cell to a file. Why?
 So ProcessPoolExecutor can import it from worker processes across platforms (not just in Google Colab)'''

import heapq
import time
from dataclasses import dataclass, field
import numpy as np
import pandas as pd

MONTH_LEN = 30.44      # days/month (~365/12)
N_MONTHS = 36           # 36-month observation window
DEFAULT_T = 365 * 3     # 1,095 days = 3 years


class Person:
    __slots__ = (
        "amIhomeless", "num_eps", "num_months", "num_undetected_months",
        "consecutive", "episodic", "months_homeless", "months_detected",
        "this_episode_months", "ep_and_cons", "p_exit", "ever_episodic",
        "ever_consecutive", "ever_homeless", "index", "next_event"
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
        self.next_event = "homeless"

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


def consecutive_test(months_homeless, this_month, fix_known_bugs=True):
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

def detect_consec(person, this_month):
    low = max(this_month - 13, -1)
    counter = 0
    for x in range(this_month, low, -1):
        if person.months_homeless[x] and person.months_detected[x]:
            counter +=1
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
    # times: list = field(default_factory=list)
    # now_homeless_ts: list = field(default_factory=list)
    # ever_homeless_ts: list = field(default_factory=list)
    # now_consec_ts: list = field(default_factory=list)
    # consecutive_ts: list = field(default_factory=list)
    # now_episodic_ts: list=field(default_factory=list)
    # episodic_ts: list = field(default_factory=list)


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
                          seed=None, fix_known_bugs=False):

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
            if rng.random() < p_u:
                person.next_event = "undetected"
            else:
                person.next_event = "exit"

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

    while event_queue:
        t, idx, event = heapq.heappop(event_queue)
        print(list(x[0] for x in event_queue))
        if t >= T:
            break
        person = population[idx]

        if person.next_event == "homeless":
            if not person.amIhomeless:
                person.next_event = person.makemehomeless(t, p_u, p_e_base, rng, fix_known_bugs=fix_known_bugs)
            else:
                person.shouldIaddmonth(t, p_e_base)
                if fix_known_bugs:
                    person.next_event = "exit" if rng.random() < (person.p_exit / (p_u + person.p_exit)) else "undetected"
        elif person.next_event == "undetected" and person.amIhomeless:
            person.addundetectedmonth(t)
            person.next_event = "exit" if rng.random() < (person.p_exit / (p_u + person.p_exit)) else "undetected"
        elif person.next_event == "exit" and person.amIhomeless:
            person.exit_homelessness(t, p_e_base, fix_known_bugs=fix_known_bugs)
            person.next_event = "homeless"

        person.num_months = sum(person.months_homeless[: person._month_index(t) + 1])

        tau = _draw_tau(rng, p_h, p_u, person.p_exit, person.next_event)
        heapq.heappush(event_queue, (t + tau, idx, person.next_event))

    last_month = min(int((T - 1) // MONTH_LEN), N_MONTHS - 1)
    for person in population:
        person.num_months = sum(person.months_homeless[: last_month + 1])
        person.amIchronic(T, fix_known_bugs=fix_known_bugs)

    n_consecutive = sum(1 for p in population if p.ever_consecutive)   # Total across all unique Persons for all months
    n_episodic = sum(1 for p in population if p.ever_episodic)
    n_ever_homeless = sum(1 for p in population if p.ever_homeless)
    n_chronic = n_consecutive + n_episodic
    # n_undetected, frac_undetected = _find_undetected_chronic_vectorized(population)

    monthly_results = pd.DataFrame(index = range(0, 36), columns = ['Now Homeless', 'Now Chronic', 'Now Episodic', 'Now Consecutive',
                                              'Now Episodic Undetected', 'Now Consecutive Undetected'])
    for month in range(0, 36):
        now_h = 0
        now_chr = 0
        now_ep = 0
        now_consec = 0
        now_ep_und = 0
        now_consec_und = 0
        for person in population:
            if person.months_homeless[month]:
                now_h +=1
            if month >= 12:
                if person.amIchronic[month]:
                    now_chr +=1
                if consecutive_test(person.months_homeless, month):
                    now_consec +=1
                    if detect_consec(person, month) == False:
                        now_consec_und +=1
                if episodic_test(person.months_homeless, month):
                    now_ep +=1
                    if detect_episodic(person, month) == False:
                        now_ep_und +=1
        monthly_results.add([now_h, now_chr, now_ep, now_consec, now_ep_und, now_consec_und])


    wall = time.perf_counter() - _start
    return RunResult(
        p_h=p_h, p_e=p_e_base, p_u=p_u, sim=sim_id,
        n_consecutive=n_consecutive,
        n_episodic=n_episodic,
        n_chronic=n_chronic,
        n_ever_homeless=n_ever_homeless,
        wall_time_sec=wall
    )
