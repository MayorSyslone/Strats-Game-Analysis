import math
import random
from collections import Counter
from itertools import combinations, product

import numpy as np
import matplotlib.pyplot as plt

ALL_CARDS = [f"{suit}{rank}" for suit in ['s', 'd', 'h', 'c'] for rank in range(1, 14)] + ['joker1', 'joker2']


# =====================================================================
# Shared game helpers (unchanged from your original)
# =====================================================================

def split_deck_with_discard(num_decks=2):
    suits = ['s', 'd', 'h', 'c']
    cards = [f"{suit}{rank}" for suit in suits for rank in range(1, 14)]
    cards.extend(['joker1', 'joker2'])

    deck_array = np.array(cards)
    np.random.shuffle(deck_array)

    total_cards = len(deck_array)
    cards_per_deck = total_cards // num_decks
    total_distributed = num_decks * cards_per_deck

    distributed_cards, discarded_cards = np.split(deck_array, [total_distributed])
    sub_decks = np.split(distributed_cards, num_decks)

    return [deck.tolist() for deck in sub_decks], discarded_cards.tolist()


def get_card_value(card):
    if 'joker' in card:
        return 'joker'
    return int(card[1:])


def calculate_deck_value(deck):
    """Calculates total point value of a deck before play, treating Jokers as 0."""
    total = 0
    for card in deck:
        val = get_card_value(card)
        if val != 'joker':
            total += val
    return total


def evaluate_combination(cards_combo, target_limit):
    has_joker = any(get_card_value(c) == 'joker' for c in cards_combo)
    limit = 15 if has_joker else target_limit

    total = 0
    for card in cards_combo:
        val = get_card_value(card)
        if val == 'joker':
            continue
        total += val

    if total > limit:
        return -1, has_joker
    return total, has_joker


def num_to_keep(slice_size, num_winners):
    """Winner keeps half their winnings. A sole winner rounds UP; tied winners round down."""
    if num_winners == 1:
        return math.ceil(slice_size / 2)
    return slice_size // 2


def lowest_card(hand):
    return min(hand, key=lambda c: 0 if 'joker' in c else get_card_value(c))


def best_combo(hand, target):
    """Highest-scoring legal play; ties broken by fewest cards."""
    valid_plays = []
    for r in range(1, len(hand) + 1):
        for combo in combinations(hand, r):
            score, _ = evaluate_combination(combo, target)
            if score != -1:
                valid_plays.append((score, combo))
    if not valid_plays:
        return [lowest_card(hand)]
    top = max(score for score, _ in valid_plays)
    return list(min((combo for score, combo in valid_plays if score == top), key=len))


# =====================================================================
# Original perfect-play simulation (round-up fix applied)
# =====================================================================

def choose_best_play(hand, max_possible_opponent_score, base_target=20):
    valid_plays = []

    for r in range(1, len(hand) + 1):
        for combo in combinations(hand, r):
            score, has_joker = evaluate_combination(combo, base_target)
            if score != -1:
                valid_plays.append((score, combo))

    if not valid_plays:
        return [lowest_card(hand)]

    max_hand_score = max(play[0] for play in valid_plays)

    if max_hand_score >= max_possible_opponent_score:
        best_combos = [combo for score, combo in valid_plays if score == max_hand_score]
        return list(min(best_combos, key=len))
    else:
        return [lowest_card(hand)]


def simulate_single_game():
    decks, discard_pile = split_deck_with_discard(num_decks=2)

    # Calculate initial disparity (Player 1 value - Player 2 value)
    p1_val = calculate_deck_value(decks[0])
    p2_val = calculate_deck_value(decks[1])
    disparity = p1_val - p2_val

    while sum(1 for d in decks if len(d) > 0) > 1:
        active_indices = [i for i, d in enumerate(decks) if len(d) > 0]
        drawn_hands = {}

        for idx in active_indices:
            num_to_draw = min(3, len(decks[idx]))
            drawn_hands[idx] = [decks[idx].pop(0) for _ in range(num_to_draw)]

        global_joker_present = any(
            any(get_card_value(c) == 'joker' for c in hand)
            for hand in drawn_hands.values()
        )
        base_target = 15 if global_joker_present else 21

        max_scores = {}
        for idx, hand in drawn_hands.items():
            possible_scores = []
            for r in range(1, len(hand) + 1):
                for combo in combinations(hand, r):
                    score, _ = evaluate_combination(combo, base_target)
                    if score != -1:
                        possible_scores.append(score)
            max_scores[idx] = max(possible_scores) if possible_scores else -1

        played_cards = {}
        round_scores = {}

        for idx, hand in drawn_hands.items():
            other_max = max([score for i, score in max_scores.items() if i != idx], default=-1)
            chosen = choose_best_play(hand, other_max, base_target)

            played_cards[idx] = chosen
            unplayed = [c for c in hand if c not in chosen]
            decks[idx] = decks[idx] + unplayed  # unused cards go to the bottom

            score, _ = evaluate_combination(chosen, base_target)
            round_scores[idx] = score

        valid_scores = {idx: sc for idx, sc in round_scores.items() if sc != -1}
        table_pool = [card for cards in played_cards.values() for card in cards]

        if valid_scores:
            winning_score = max(valid_scores.values())
            winners = [idx for idx, sc in valid_scores.items() if sc == winning_score]

            cards_per_winner = len(table_pool) // len(winners)
            random.shuffle(table_pool)

            for w in winners:
                won_slice = table_pool[:cards_per_winner]
                table_pool = table_pool[cards_per_winner:]

                kept = random.sample(won_slice, num_to_keep(len(won_slice), len(winners)))
                decks[w].extend(kept)
                discard_pile.extend(c for c in won_slice if c not in kept)

            discard_pile.extend(table_pool)
        else:
            discard_pile.extend(table_pool)

    active_winners = [i for i, d in enumerate(decks) if len(d) > 0]
    if len(active_winners) == 1:
        winner_idx = active_winners[0]
        winning_card_count = len(decks[winner_idx])
        # Positive if Player 1 won, negative if Player 2 won
        win_margin = winning_card_count if winner_idx == 0 else -winning_card_count
    else:
        win_margin = 0  # Draw

    return disparity, win_margin


# =====================================================================
# Confidence-based players
# =====================================================================

# Lookup table of a hand's best legal score: MAX_TABLE[target_idx, a, b, c]
# target_idx 0 -> target 15, 1 -> target 21. Card codes: 0 = joker, 1-13 = rank, 14 = empty slot.
EMPTY = 14


def _build_max_table():
    table = np.zeros((2, 15, 15, 15), dtype=np.int16)
    for t_idx, target in enumerate((15, 21)):
        for a in range(15):
            for b in range(15):
                for c in range(15):
                    vals = [0 if x in (0, EMPTY) else x for x in (a, b, c)]
                    best = -1
                    for r in range(1, 4):
                        for combo in combinations(vals, r):
                            s = sum(combo)
                            if best < s <= target:
                                best = s
                    table[t_idx, a, b, c] = best
    return table


MAX_TABLE = _build_max_table()


def card_code(card):
    return 0 if 'joker' in card else int(card[1:])


def hand_codes(cards):
    codes = [card_code(c) for c in cards]
    return codes + [EMPTY] * (3 - len(codes))


class PlayerKnowledge:
    """
    Everything one player knows about the cards they can't see.

    - seen:   every card that has been in this player's hand or played by anyone.
              Any card NOT seen is equally likely to be in any 'orig' (never-touched) slot.
    - groups: for each batch of winnings the opponent kept, the cards it could contain.
              The opponent kept a random subset of that pool, minus anything this player kept
              from the same pool (ties), minus anything later seen played again.

    Each card in a deck carries a public tag: 'orig' for the starting deal, or a group id
    for winnings added to the bottom. The tags are public because the order of draws,
    returns and pick-ups is visible; card identities stay hidden.
    """

    def __init__(self):
        self.seen = set()
        self.groups = {}

    def see_own_hand(self, cards):
        self.seen.update(cards)

    def see_played(self, cards):
        self.seen.update(cards)
        for candidates in self.groups.values():
            candidates.difference_update(cards)

    def see_opponent_keep(self, tag, pool, my_kept):
        self.groups[tag] = set(pool) - set(my_kept)


def opponent_hand_slots(opp_drawn, knowledge):
    """
    For the opponent's drawn hand, a list of (candidate card codes, how many cards come from them).

    opp_drawn is the opponent's drawn hand as (card, tag) pairs. Only the tags are read
    (where in the deck each card came from, which is public); the cards themselves are hidden.
    Jokers are hidden too, so they are candidates like any other card.

    - 'orig' cards: any card this player has never seen.
    - cards from a prize pool the opponent kept: any card left in that pool's candidate list.
    """
    slots = []
    for tag, m in Counter(tag for _, tag in opp_drawn).items():
        if tag == 'orig':
            candidates = [c for c in ALL_CARDS if c not in knowledge.seen]
        else:
            candidates = list(knowledge.groups[tag])
        slots.append((np.array([card_code(c) for c in candidates]), m))
    return slots


def possible_opponent_hands(opp_drawn, knowledge, max_hands, rng):
    """
    Possible opponent hands as card codes (rows x 3, EMPTY-padded), each equally likely.

    If the number of possible hands is at most max_hands, returns every one of them (exact).
    Otherwise returns max_hands hands chosen at random (an estimate).
    """
    slots = opponent_hand_slots(opp_drawn, knowledge)
    total = math.prod(math.comb(len(codes), m) for codes, m in slots)

    if total <= max_hands:
        # Exact: every combination from each source, combined with every combination from the others
        per_slot = [list(combinations(codes.tolist(), m)) for codes, m in slots]
        rows = [sum(parts, ()) for parts in product(*per_slot)]
        hands = np.full((len(rows), 3), EMPTY, dtype=np.int64)
        for r, row in enumerate(rows):
            hands[r, :len(row)] = row
        return hands

    # Too many to check: sample max_hands of them
    hands = np.full((max_hands, 3), EMPTY, dtype=np.int64)
    col = 0
    for codes, m in slots:
        # m distinct cards per sample, uniformly at random
        picks = np.argpartition(rng.random((max_hands, len(codes))), m - 1, axis=1)[:, :m]
        hands[:, col:col + m] = codes[picks]
        col += m
    return hands


def estimate_win_chance(my_hand, opp_drawn, knowledge, max_hands=2000, rng=None):
    """
    Returns (win_chance, best_play): the play from my hand most likely to win this round,
    and its chance (0-1) of beating the opponent's best play. A tie counts as half a win.

    Neither player knows whether the opponent holds a joker. For each possible opponent hand,
    the cap is 15 if that possible hand or mine has a joker, otherwise 21. A play that busts
    under that cap loses that possibility, and the opponent's best play is worked out
    under the same cap.
    """
    rng = rng or np.random.default_rng()
    opp = possible_opponent_hands(opp_drawn, knowledge, max_hands, rng)

    i_have_joker = any('joker' in c for c in my_hand)
    opp_has_joker = (opp == 0).any(axis=1)
    cap = np.where(i_have_joker | opp_has_joker, 15, 21)
    t_idx = (cap == 21).astype(np.int64)
    opp_max = MAX_TABLE[t_idx, opp[:, 0], opp[:, 1], opp[:, 2]]

    best_chance, best_play = -1.0, None
    for r in range(1, len(my_hand) + 1):
        for combo in combinations(my_hand, r):
            total = sum(card_code(c) for c in combo)
            my_score = np.where(total <= cap, total, -1)
            chance = float(np.mean(my_score > opp_max) + 0.5 * np.mean(my_score == opp_max))
            # Higher chance wins; on a tie, keep the play that uses fewer cards
            if chance > best_chance + 1e-12:
                best_chance, best_play = chance, list(combo)
    return best_chance, best_play


def decide_to_go(win_chance, confidence):
    """2*win_chance + confidence - 2: above 0 go for it, below 0 don't, exactly 0 coin flip."""
    decision = 2 * win_chance + confidence - 2
    if math.isclose(decision, 0, abs_tol=1e-9):
        return random.random() < 0.5
    return decision > 0


def simulate_confidence_game(confidence_p1=0.5, confidence_p2=0.5, max_hands=2000, rng=None):
    """
    One game between two players who estimate their round win chance from what they've seen,
    then go for the round (best play) or concede (lowest card) based on their confidence.

    Returns (disparity, win_margin) in the same form as simulate_single_game.
    """
    rng = rng or np.random.default_rng()
    raw_decks, discard_pile = split_deck_with_discard(num_decks=2)
    disparity = calculate_deck_value(raw_decks[0]) - calculate_deck_value(raw_decks[1])

    decks = [[(card, 'orig') for card in d] for d in raw_decks]
    knowledge = [PlayerKnowledge(), PlayerKnowledge()]
    confidences = [confidence_p1, confidence_p2]
    round_no = 0

    while all(len(d) > 0 for d in decks):
        round_no += 1

        drawn = {}
        for i in (0, 1):
            n = min(3, len(decks[i]))
            drawn[i], decks[i] = decks[i][:n], decks[i][n:]
            knowledge[i].see_own_hand(card for card, _ in drawn[i])

        joker_present = any('joker' in card for hand in drawn.values() for card, _ in hand)
        target = 15 if joker_present else 21

        # Both players decide simultaneously, without seeing the other's hand.
        # The real cap (target) is only applied when the round is scored.
        played = {}
        for i in (0, 1):
            hand = [card for card, _ in drawn[i]]
            win_chance, play = estimate_win_chance(hand, drawn[1 - i], knowledge[i], max_hands, rng)
            if decide_to_go(win_chance, confidences[i]):
                played[i] = play
            else:
                played[i] = [lowest_card(hand)]

        for i in (0, 1):
            unplayed = [entry for entry in drawn[i] if entry[0] not in played[i]]
            decks[i] = decks[i] + unplayed  # unused cards go to the bottom (tags travel with them)

        table_pool = played[0] + played[1]
        for k in knowledge:
            k.see_played(table_pool)

        scores = {i: evaluate_combination(played[i], target)[0] for i in (0, 1)}
        valid_scores = {i: s for i, s in scores.items() if s != -1}

        if not valid_scores:
            discard_pile.extend(table_pool)
            continue

        winning_score = max(valid_scores.values())
        winners = [i for i, s in valid_scores.items() if s == winning_score]

        pool = table_pool[:]
        random.shuffle(pool)
        cards_per_winner = len(pool) // len(winners)
        kept_by = {}

        for w in winners:
            won_slice, pool = pool[:cards_per_winner], pool[cards_per_winner:]
            kept = random.sample(won_slice, num_to_keep(len(won_slice), len(winners)))
            kept_by[w] = kept
            decks[w].extend((card, (round_no, w)) for card in kept)
            discard_pile.extend(c for c in won_slice if c not in kept)
        discard_pile.extend(pool)

        # Each player learns how many cards the opponent kept and from which pool
        for w in winners:
            observer = 1 - w
            knowledge[observer].see_opponent_keep((round_no, w), table_pool, kept_by.get(observer, []))

    survivors = [i for i in (0, 1) if decks[i]]
    if len(survivors) == 1:
        w = survivors[0]
        win_margin = len(decks[w]) if w == 0 else -len(decks[w])
    else:
        win_margin = 0  # Draw

    return disparity, win_margin


# Settings
MODE = "confidence"        
NUM_SIMULATIONS = 1000
CONFIDENCE_P1 = 1        # 0-1
CONFIDENCE_P2 = 0.4        # 0-1
MAX_HANDS = 2000           # most opponent hands checked per decision: exact when there are fewer possible hands, sampled when there are more


def run_simulations(num_simulations, mode="confidence", confidence_p1=0.5, confidence_p2=0.5, max_hands=2000):
    
    disparities = []
    win_margins = []

    print(f"Running {num_simulations} simulations ({mode})...")
    for _ in range(num_simulations):
        if mode == "perfect":
            d, w = simulate_single_game()
        else:
            d, w = simulate_confidence_game(confidence_p1, confidence_p2, max_hands)
        disparities.append(d)
        win_margins.append(w)

    return np.array(disparities), np.array(win_margins)


def plot_results(disparities, win_margins, subtitle=""):
    """Scatter of initial deck value disparity vs final win margin, one point per game, with trendline."""
    num_simulations = len(disparities)

    plt.figure(figsize=(10, 6))
    plt.scatter(disparities, win_margins, alpha=0.4, color='b', edgecolors='none', label='Game Outcome')

    if num_simulations > 1:
        m, b = np.polyfit(disparities, win_margins, 1)
        x_vals = np.array([min(disparities), max(disparities)])
        plt.plot(x_vals, m * x_vals + b, color='red', linewidth=2, label=f'Trendline (y = {m:.2f}x + {b:.2f})')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    title = f"Initial Deck Value Disparity vs. Final Win Margin ({num_simulations} Runs)"
    if subtitle:
        title += f"\n{subtitle}"
    plt.title(title, fontsize=14)
    plt.xlabel("Initial Deck Value Disparity (Player 1 - Player 2)", fontsize=12)
    plt.ylabel("Final Win Margin in Cards (Positive = P1 Win, Negative = P2 Win)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    disparities, win_margins = run_simulations(
        NUM_SIMULATIONS, MODE, CONFIDENCE_P1, CONFIDENCE_P2, MAX_HANDS
    )
    if MODE == "perfect":
        subtitle = "Perfect play"
    else:
        subtitle = f"P1 confidence = {CONFIDENCE_P1}, P2 confidence = {CONFIDENCE_P2}"
    plot_results(disparities, win_margins, subtitle)