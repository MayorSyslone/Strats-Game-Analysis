import numpy as np
import random
from itertools import combinations
import matplotlib.pyplot as plt

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

def choose_best_play(hand, max_possible_opponent_score, base_target=21):
    valid_plays = []
    
    for r in range(1, len(hand) + 1):
        for combo in combinations(hand, r):
            score, has_joker = evaluate_combination(combo, base_target)
            if score != -1:
                valid_plays.append((score, combo))
                
    if not valid_plays:
        lowest_card = min(hand, key=lambda c: 0 if 'joker' in c else get_card_value(c))
        return [lowest_card]
    
    max_hand_score = max(play[0] for play in valid_plays)
    
    if max_hand_score >= max_possible_opponent_score:
        best_combos = [combo for score, combo in valid_plays if score == max_hand_score]
        best_combo = min(best_combos, key=len)
        return list(best_combo)
    else:
        lowest_card = min(hand, key=lambda c: 0 if 'joker' in c else get_card_value(c))
        return [lowest_card]

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
            decks[idx] = unplayed + decks[idx]
            
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
                
                num_to_keep = len(won_slice) // 2
                kept = random.sample(won_slice, num_to_keep)
                decks[w].extend(kept)
                
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

# --- Main Batch Simulation & Plotting ---
def run_and_plot_simulations(num_simulations=500):
    disparities = []
    win_margins = []

    print(f"Running {num_simulations} simulations...")
    for _ in range(num_simulations):
        d, w = simulate_single_game()
        disparities.append(d)
        win_margins.append(w)

    disparities = np.array(disparities)
    win_margins = np.array(win_margins)

    # Plotting
    plt.figure(figsize=(10, 6))
    
    # Scatter plot with slight transparency for overlapping points
    plt.scatter(disparities, win_margins, alpha=0.4, color='b', edgecolors='none', label='Game Outcome')
    
    # Trendline (Linear Fit)
    if len(disparities) > 1:
        m, b = np.polyfit(disparities, win_margins, 1)
        x_vals = np.array([min(disparities), max(disparities)])
        plt.plot(x_vals, m * x_vals + b, color='red', linewidth=2, label=f'Trendline (y = {m:.2f}x + {b:.2f})')

    # Reference lines
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.7)

    plt.title(f"Initial Deck Value Disparity vs. Final Win Margin ({num_simulations} Runs)", fontsize=14)
    plt.xlabel("Initial Deck Value Disparity (Player 1 - Player 2)", fontsize=12)
    plt.ylabel("Final Win Margin in Cards (Positive = P1 Win, Negative = P2 Win)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# Run 500 game simulations and plot results
run_and_plot_simulations(num_simulations=2000)