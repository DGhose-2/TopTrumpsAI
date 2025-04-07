# main.py
"""
Main script using corrected GameEngine and Agents following strict info constraints.
Passes only `my_initial_card_ids` (set) and `full_game_deck_ref_df` at init.
"""
import json, time, random, pprint
from typing import List, Dict, Any, Optional, Set # Added Set
import numpy as np, pandas as pd
from card_utils import generate_decks
from agents import create_agent, Agent
from game_engine import GameEngine

# ==============================
# --- Simulation Runner ---
# ==============================
def run_trials(trials_config: List[List[Any]], n_fields: int = 5, max_tricks_per_game: int = 5000, goliath_p1_debug: bool = False, goliath_p2_debug: bool = False, engine_debug: bool = False) -> Dict[str, Dict[str, Any]]:
    results = {}; total_start_time = time.time()
    for i, arg_set in enumerate(trials_config):
        print(f"\n--- Trial Set {i+1}/{len(trials_config)} ---")
        set_start_time = time.time()
        try:
            p1s, p2s, nc, nt = arg_set[:4]; p1rmf=arg_set[4] if len(arg_set)>4 else None; p2rmf=arg_set[5] if len(arg_set)>5 else None
            p1m=arg_set[6] if len(arg_set)>6 and arg_set[6] is not None else 0.5; p2m=arg_set[7] if len(arg_set)>7 and arg_set[7] is not None else 0.5
        except Exception as e: print(f"Err unpack {i+1}: {e}. Skip."); continue
        print(f"Config: P1={p1s}, P2={p2s}, C={nc}, F={n_fields}, T={nt}")
        details = []
        if p1s.lower()=='randmaxer' and p1rmf is not None: details.append(f"P1_RMax{p1rmf:.2f}")
        if p2s.lower()=='randmaxer' and p2rmf is not None: details.append(f"P2_RMax{p2rmf:.2f}")
        if p1s.lower() in ['expert','goliath']: details.append(f"P1_M{p1m:.2f}")
        if p2s.lower() in ['expert','goliath']: details.append(f"P2_M{p2m:.2f}")
        if goliath_p1_debug and p1s.lower()=='goliath': details.append("P1_GDBG");
        if goliath_p2_debug and p2s.lower()=='goliath': details.append("P2_GDBG");
        if engine_debug: details.append("EngDBG")
        if details: print(f"  Params: {', '.join(details)}")

        p1w=0; p2w=0; dc=0; ec=0
        for trial in range(nt):
            try: p1_deck_init, p2_deck_init, _, fields, full_ref = generate_decks(nc, n_fields)
            except Exception as e: print(f"\nErr deck gen {trial+1}: {e}. Skip."); ec+=1; continue
            try:
                p1_ids_set = set(p1_deck_init['id']); p2_ids_set = set(p2_deck_init['id'])
                player1 = create_agent(strategy=p1s, my_initial_card_ids=p1_ids_set, full_game_deck_ref_df=full_ref, randmax_frac=p1rmf, EXPert_m=p1m, goliath_debug=goliath_p1_debug if p1s.lower()=='goliath' else False, n_fields=n_fields, n_cards=nc)
                player2 = create_agent(strategy=p2s, my_initial_card_ids=p2_ids_set, full_game_deck_ref_df=full_ref, randmax_frac=p2rmf, EXPert_m=p2m, goliath_debug=goliath_p2_debug if p2s.lower()=='goliath' else False, n_fields=n_fields, n_cards=nc)
            except Exception as e: print(f"\nErr agent create {trial+1}: {e}. Skip."); ec+=1; continue
            try: game = GameEngine(player1, player2, p1_deck_init, p2_deck_init, fields, engine_debug)
            except Exception as e: print(f"\nErr engine create {trial+1}: {e}. Skip."); ec+=1; continue
            winner = None
            try: winner = game.play_game(print_tricks=False, max_tricks=max_tricks_per_game)
            except Exception as e: print(f"\nCRIT Err game play {trial+1}: {e}. Skip."); winner='error'
            if winner=='player1': print('W',end='',flush=True); p1w+=1
            elif winner=='player2': print('L',end='',flush=True); p2w+=1
            elif winner=='draw': print('D',end='',flush=True); dc+=1
            else: print('?',end='',flush=True); ec+=1
            if (trial + 1) % 50 == 0: print('|', end='', flush=True)
            if (trial + 1) % 500 == 0: print(f' [{trial+1}/{nt}]')
        set_end = time.time(); print(f"\nSet {i+1} Done ({set_end-set_start_time:.2f}s). P1W={p1w}, P2W={p2w}, D={dc}, E={ec}")
        p1ps=f"R{p1rmf:.1f}" if p1s.lower()=='randmaxer' else f"M{p1m:.1f}" if p1s.lower() in ['expert','goliath'] else ""
        p2ps=f"R{p2rmf:.1f}" if p2s.lower()=='randmaxer' else f"M{p2m:.1f}" if p2s.lower() in ['expert','goliath'] else ""
        key=f"P1_{p1s}{'('+p1ps+')' if p1ps else ''}_vs_P2_{p2s}{'('+p2ps+')' if p2ps else ''}_C{nc}_F{n_fields}_T{nt}"
        results[key]={'p1s':p1s,'p2s':p2s,'nc':nc,'nf':n_fields,'nt':nt,'p1p':{'rmf':p1rmf,'em':p1m},'p2p':{'rmf':p2rmf,'em':p2m},'p1w':p1w,'p2w':p2w,'d':dc,'e':ec,'t':round(set_end-set_start_time,2)}
        tstamp=time.strftime("%Y%m%d-%H%M%S"); res_fname=f'toptrumps_results_{tstamp}.json'
        try:
            with open(res_fname,'w') as f: json.dump(results, f, indent=4); print(f"  Progress saved: {res_fname}")
        except Exception as e: print(f" Err saving progress: {e}")
    total_end=time.time(); print(f"\n--- All Trials Done ({total_end-total_start_time:.2f}s) ---"); print("\nFinal Results:"); pprint.pprint(results)
    sum_fname=f'toptrumps_summary_{tstamp}.txt'
    try:
        with open(sum_fname,'w') as f:
            f.write(f"TopTrumps Results ({time.strftime('%Y-%m-%d %H:%M:%S')})\nTotal Dur:{total_end-total_start_time:.2f}s\n==\n")
            for k, res in results.items():
                f.write(f"{k}\n P1({res['p1s']})W:{res['p1w']}\n P2({res['p2s']})W:{res['p2w']}\n D:{res['d']}\n")
                if res['e']>0:
                    f.write(f" E:{res['e']}\n")
                    f.write(f" Tot:{res['p1w']+res['p2w']+res['d']+res['e']}/{res['nt']}\n Dur:{res['t']}s\n--\n")
        print(f"\nSummary saved: {sum_fname}")
    except Exception as e: print(f" Err saving summary: {e}")
    return results

# ==============================
# --- Example Single Game ---
# ==============================
def run_single_game(p1_strat='Goliath', p2_strat='EXPert', n_cards=30, n_fields=5, p1_m=0.5, p2_m=0.5, p1_goliath_debug=True, p2_goliath_debug=False, engine_debug=False):
    print(f"\n--- Single Game: {p1_strat} vs {p2_strat} (C={n_cards}, F={n_fields}) ---")
    try: p1_deck, p2_deck, _, fields, full_ref = generate_decks(n_cards, n_fields)
    except Exception as e: print(f"Error deck gen: {e}"); return
    try:
        p1_ids_set = set(p1_deck['id']); p2_ids_set = set(p2_deck['id']) # Get initial IDs as sets
        player1 = create_agent(strategy=p1_strat, my_initial_card_ids=p1_ids_set, full_game_deck_ref_df=full_ref, EXPert_m=p1_m, goliath_debug=p1_goliath_debug if p1_strat.lower()=='goliath' else False, n_fields=n_fields, n_cards=n_cards)
        player2 = create_agent(strategy=p2_strat, my_initial_card_ids=p2_ids_set, full_game_deck_ref_df=full_ref, EXPert_m=p2_m, goliath_debug=p2_goliath_debug if p2_strat.lower()=='goliath' else False, n_fields=n_fields, n_cards=n_cards)
    except Exception as e: print(f"Error agent create: {e}"); return
    try: game = GameEngine(player1, player2, p1_deck, p2_deck, fields, engine_debug)
    except Exception as e: print(f"Error engine create: {e}"); return
    try: winner = game.play_game(print_tricks=True, max_tricks=1000)
    except Exception as e: print(f"Error game play: {e}"); winner = "Error"
    print(f"\n--- Single Game Finished. Result: {winner.upper() if winner in ['player1','player2','draw'] else winner} ---")

# ==============================
# --- Original Trials Config Def ---
# ==============================
def define_original_trials_config() -> List[List[Any]]:
    N_cards2=50; N_trials2=100
    return [ ['Goliath','EXPert',N_cards2,N_trials2,None,None,0.5,0.5], ['EXPert','Goliath',N_cards2,N_trials2,None,None,0.5,0.5], ['EXPert','meanermax',N_cards2,N_trials2,None,None,0.5,None], ['meanermax','EXPert',N_cards2,N_trials2,None,None,None,0.5], ['meanermax','maxer',N_cards2,N_trials2,None,None,None,None], ['maxer','EXPert',N_cards2,N_trials2,None,None,None,0.5], ['EXPert','maxer',N_cards2,N_trials2,None,None,0.5,None], ['maxer','Goliath',N_cards2,N_trials2,None,None,None,0.5], ['Goliath','maxer',N_cards2,N_trials2,None,None,0.5,None], ['meanermax','EXPert',N_cards2,N_trials2,None,None,None,0.5], ['EXPert','meanermax',N_cards2,N_trials2,None,None,0.5,None], ['meanermax','Goliath',N_cards2,N_trials2,None,None,None,0.5], ['EXPert','Goliath',N_cards2,N_trials2,None,None,0.5,0.5], ['Goliath','EXPert',N_cards2,N_trials2,None,None,0.5,0.5], ['rander','rander',N_cards2,N_trials2,None,None,None,None], ['maxer','maxer',N_cards2,N_trials2,None,None,None,None], ['meanermax','meanermax',N_cards2,N_trials2,None,None,None,None], ['EXPert','EXPert',N_cards2,N_trials2,None,None,0.5,0.5], ['Goliath','Goliath',N_cards2,N_trials2,None,None,0.5,0.5] ]

# ==============================
# --- Main Execution Block ---
# ==============================
if __name__ == "__main__":
    random.seed(111); np.random.seed(111) # Example seeds

    # === Option 1: Single Game ===
    # run_single_game( p1_strat='Goliath', p2_strat='EXPert', n_cards=20, p1_goliath_debug=False, engine_debug=False)

    # === Option 2: Batch Trials ===
    print("\n" + "="*60 + "\nStarting Batch Trials...\n" + "="*60 + "\n")
    trial_configs = define_original_trials_config()
    # Reduce N_trials2 inside define_original_trials_config() for faster testing
    results = run_trials(trials_config=trial_configs, n_fields=5, max_tricks_per_game=10000) # Increased max_tricks

    print("\nScript finished.")