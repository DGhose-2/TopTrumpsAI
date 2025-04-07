# main.py
"""
Main script to run Top Trumps simulations or single games.

Uses the optimized GameEngine and strict Agent implementations.
Handles deck generation, agent creation, game execution, and results reporting.
"""
import os
import json
import time
import random
import pprint # For printing results nicely
from typing import List, Dict, Any, Optional, Set
import numpy as np
import pandas as pd
# deque is used internally by engine, not directly needed here
# from collections import deque

# Assuming card_utils.py, agents.py, game_engine.py are accessible
from card_utils import generate_decks
from agents import create_agent, Agent # Import base class if needed for type hints
from game_engine import GameEngine # Uses the optimized engine

# ==============================
# --- Simulation Runner ---
# ==============================

def run_trials(trials_config: List[List[Any]],
               number_of_fields: int = 5,
               max_tricks_per_game: int = 5000,
               goliath_player1_debug: bool = False,
               goliath_player2_debug: bool = False,
               engine_debug_flag: bool = False
               ) -> Dict[str, Dict[str, Any]]:
    """
    Runs multiple trials (games) based on configurations provided.

    Args:
        trials_config: A list where each item is a list defining a trial set:
                       [p1_strategy (str), p2_strategy (str), num_cards (int), num_trials (int),
                        p1_randmax_frac (Optional[float]), p2_randmax_frac (Optional[float]),
                        p1_expert_m (float, default 0.5), p2_expert_m (float, default 0.5)]
                       Uses descriptive names internally now.
        number_of_fields: Number of fields (attributes) for the cards in all trials.
        max_tricks_per_game: Max tricks allowed before game declared draw by count.
        goliath_player1_debug: Enable verbose logging for P1 if it's Goliath.
        goliath_player2_debug: Enable verbose logging for P2 if it's Goliath.
        engine_debug_flag: Enable verbose logging for the GameEngine.

    Returns:
        A dictionary containing the aggregated results for each trial configuration key.
    """
    # Dictionary to store results for all configurations run in this batch
    simulation_results: Dict[str, Dict[str, Any]] = {}
    total_start_time: float = time.time()
    # Timestamp used for saving results progressively and finally
    last_results_timestamp: str = ""

    # --- Iterate Through Each Configuration Set ---
    for config_index, trial_config_set in enumerate(trials_config):
        print(f"\n--- Running Trial Set {config_index+1}/{len(trials_config)} ---")
        set_start_time: float = time.time()

        # --- Unpack Arguments from the configuration list ---
        try:
            # Use descriptive names for unpacked variables
            player1_strategy_name: str = trial_config_set[0]
            player2_strategy_name: str = trial_config_set[1]
            number_of_cards_in_game: int = trial_config_set[2]
            number_of_trials_to_run: int = trial_config_set[3]
            # Optional parameters with default values if not provided
            player1_randmax_fraction: Optional[float] = trial_config_set[4] if len(trial_config_set) > 4 else None
            player2_randmax_fraction: Optional[float] = trial_config_set[5] if len(trial_config_set) > 5 else None
            # Use descriptive name for M-value parameter expected by agents.py/create_agent
            player1_expert_m_value: float = trial_config_set[6] if len(trial_config_set) > 6 and trial_config_set[6] is not None else 0.5
            player2_expert_m_value: float = trial_config_set[7] if len(trial_config_set) > 7 and trial_config_set[7] is not None else 0.5
        except IndexError:
            # Handle improperly formatted configuration lists
            print(f"Error: Trial config set {config_index+1} incomplete: {trial_config_set}. Skipping.")
            continue # Skip to the next configuration set
        except Exception as error:
            # Catch any other errors during unpacking
            print(f"Error unpacking arguments for trial set {config_index+1}: {error}. Skipping.")
            continue

        # --- Print Configuration Details ---
        print(f"Config: P1={player1_strategy_name}, P2={player2_strategy_name}, "
              f"Cards={number_of_cards_in_game}, Fields={number_of_fields}, Trials={number_of_trials_to_run}")
        # Display specific parameters if they are relevant
        parameter_details: List[str] = []
        if player1_strategy_name.lower() == 'randmaxer' and player1_randmax_fraction is not None:
            parameter_details.append(f"P1_RandMaxFrac={player1_randmax_fraction:.2f}")
        if player2_strategy_name.lower() == 'randmaxer' and player2_randmax_fraction is not None:
            parameter_details.append(f"P2_RandMaxFrac={player2_randmax_fraction:.2f}")
        if player1_strategy_name.lower() in ['expert', 'goliath']:
            parameter_details.append(f"P1_ExpertM={player1_expert_m_value:.2f}")
        if player2_strategy_name.lower() in ['expert', 'goliath']:
            parameter_details.append(f"P2_ExpertM={player2_expert_m_value:.2f}")
        # Display debug flags if enabled
        if goliath_player1_debug and player1_strategy_name.lower() == 'goliath':
            parameter_details.append("P1_GoliathDBG")
        if goliath_player2_debug and player2_strategy_name.lower() == 'goliath':
            parameter_details.append("P2_GoliathDBG")
        if engine_debug_flag:
            parameter_details.append("EngineDBG")
        # Print the collected parameter details if any exist
        if parameter_details:
            print(f"  Params: {', '.join(parameter_details)}")

        # --- Initialize Counters for this Specific Trial Set ---
        player1_wins: int = 0
        player2_wins: int = 0
        draw_count: int = 0
        error_count: int = 0 # Count games that failed or had errors

        # --- Run Individual Trials (Games) within this Set ---
        for trial_number in range(number_of_trials_to_run):
            # --- 1. Generate Decks and Reference Data ---
            try:
                # Unpack the 4 items returned by the updated generate_decks
                p1_initial_deck_df, p2_initial_deck_df, field_names_list, full_deck_reference = \
                    generate_decks(number_of_cards_in_game, number_of_fields)

                # Check that the required reference deck was generated successfully
                if full_deck_reference is None:
                    raise ValueError("Full reference deck is required but was not generated by generate_decks.")
            except Exception as error:
                # Handle errors during deck generation
                print(f"\nError generating decks trial {trial_number+1}: {error}. Skipping.")
                error_count += 1
                continue # Skip this trial and proceed to the next

            # --- 2. Create Agent Instances ---
            try:
                # Get initial IDs as sets - required for agent initialization
                p1_initial_ids_set: Set[str] = set(p1_initial_deck_df['id'])
                p2_initial_ids_set: Set[str] = set(p2_initial_deck_df['id'])

                # Create Player 1 Agent using the factory function
                player1_agent_object: Agent = create_agent(
                    strategy=player1_strategy_name,
                    my_initial_card_ids=p1_initial_ids_set,              # Pass agent's own initial IDs
                    full_game_deck_ref_df=full_deck_reference,   # Pass full reference deck
                    randmax_fraction=player1_randmax_fraction,   # Pass RandMaxer param
                    EXPert_m_value=player1_expert_m_value,       # Pass M-value param
                    goliath_debug_flag=goliath_player1_debug if player1_strategy_name.lower()=='goliath' else False, # Pass Goliath debug flag
                    number_of_fields=number_of_fields,         # Pass context
                    number_of_cards=number_of_cards_in_game    # Pass context
                )
                # Create Player 2 Agent
                player2_agent_object: Agent = create_agent(
                    strategy=player2_strategy_name,
                    my_initial_card_ids=p2_initial_ids_set,
                    full_game_deck_ref_df=full_deck_reference,
                    randmax_fraction=player2_randmax_fraction,
                    EXPert_m_value=player2_expert_m_value,
                    goliath_debug_flag=goliath_player2_debug if player2_strategy_name.lower()=='goliath' else False,
                    number_of_fields=number_of_fields,
                    number_of_cards=number_of_cards_in_game
                )
            except Exception as error:
                # Handle errors during agent creation
                print(f"\nError creating agents trial {trial_number+1}: {type(error).__name__} - {error}. Skipping.")
                error_count += 1
                continue # Skip this trial

            # --- 3. Create Game Engine Instance ---
            try:
                # Engine needs initial DFs (to get IDs), field names, and the full reference deck
                game_engine_instance: GameEngine = GameEngine(
                    player1_agent=player1_agent_object,
                    player2_agent=player2_agent_object,
                    initial_deck1_df=p1_initial_deck_df,
                    initial_deck2_df=p2_initial_deck_df,
                    field_names=field_names_list,
                    full_game_ref_deck=full_deck_reference, # Engine needs reference deck
                    engine_debug=engine_debug_flag
                )
            except Exception as error:
                # Handle errors during engine creation
                print(f"\nError creating GameEngine trial {trial_number+1}: {type(error).__name__} - {error}. Skipping.")
                error_count += 1
                continue # Skip trial

            # --- 4. Run the Game ---
            game_winner_string: Optional[str] = None
            try:
                 # Execute the full game simulation
                game_winner_string = game_engine_instance.play_game(
                    print_tricks=False, # Set to True to see turn-by-turn details (slow)
                    max_tricks=max_tricks_per_game
                )
            except Exception as error:
                # Catch unexpected errors during game execution
                print(f"\nCRITICAL Error during game play trial {trial_number+1}: {type(error).__name__} - {error}. Recording as error.")
                game_winner_string = 'error' # Mark the outcome as an error

            # --- 5. Record Game Result ---
            if game_winner_string == 'player1':
                print('W', end='', flush=True) # Progress indicator for P1 win
                player1_wins += 1
            elif game_winner_string == 'player2':
                print('L', end='', flush=True) # Progress indicator for P1 Loss (P2 Win)
                player2_wins += 1
            elif game_winner_string == 'draw':
                print('D', end='', flush=True) # Progress indicator for Draw
                draw_count += 1
            else: # Handles 'error' or any other unexpected return value
                print('?', end='', flush=True) # Indicator for error/unknown outcome
                error_count += 1

            # --- Progress Markers for Console Output ---
            current_trial_index = trial_number + 1
            if current_trial_index % 50 == 0:
                print('|', end='', flush=True) # Major marker every 50 trials
            # Less frequent newline marker with count
            if current_trial_index % 1000 == 0 or current_trial_index == number_of_trials_to_run:
                print(f' [{current_trial_index}/{number_of_trials_to_run}]')
            # Smaller marker more frequently
            elif current_trial_index % 250 == 0:
                 print('.', end='', flush=True)


        # --- End of Trial Loop for this Configuration Set ---
        set_end_time: float = time.time()
        set_duration: float = set_end_time - set_start_time
        print(f"\nTrial Set {config_index+1} Complete ({set_duration:.2f}s)")
        print(f"  Results: P1 Wins={player1_wins}, P2 Wins={player2_wins}, Draws={draw_count}, Errors={error_count} (Total Run={number_of_trials_to_run})")

        # --- Store Results for this Configuration ---
        # Create a descriptive key based on the configuration parameters
        p1_param_desc = f"R{player1_randmax_fraction:.1f}" if player1_strategy_name.lower() == 'randmaxer' else f"M{player1_expert_m_value:.1f}" if player1_strategy_name.lower() in ['expert','goliath'] else ""
        p2_param_desc = f"R{player2_randmax_fraction:.1f}" if player2_strategy_name.lower() == 'randmaxer' else f"M{player2_expert_m_value:.1f}" if player2_strategy_name.lower() in ['expert','goliath'] else ""
        configuration_key = (f"P1_{player1_strategy_name}{'('+p1_param_desc+')' if p1_param_desc else ''}_vs_"
                             f"P2_{player2_strategy_name}{'('+p2_param_desc+')' if p2_param_desc else ''}_"
                             f"C{number_of_cards_in_game}_F{number_of_fields}_T{number_of_trials_to_run}")

        # Store detailed results in the main results dictionary
        simulation_results[configuration_key] = {
            'player1_strategy': player1_strategy_name,
            'player2_strategy': player2_strategy_name,
            'number_cards': number_of_cards_in_game,
            'number_fields': number_of_fields,
            'number_trials': number_of_trials_to_run,
            'player1_params': {'randmax_fraction': player1_randmax_fraction, 'expert_m_value': player1_expert_m_value},
            'player2_params': {'randmax_fraction': player2_randmax_fraction, 'expert_m_value': player2_expert_m_value},
            'player1_wins': player1_wins,
            'player2_wins': player2_wins,
            'draws': draw_count,
            'errors': error_count,
            'duration_seconds': round(set_duration, 2)
        }

        # --- Save Results Progressively (after each set) ---
        # Generate timestamp for unique filenames
        # Note: Using current time: Sunday, April 6, 2025 at 7:37:58 AM BST
        last_results_timestamp = time.strftime("%Y%m%d") # e.g., 20250406
        results_filename = f'results/progressive/toptrumps_results_{last_results_timestamp}.json'
        try:
            # make parent folders for writing file, if needed
            dirpath = os.path.dirname(results_filename)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(results_filename, 'w') as json_file:
                # Write the accumulated results dictionary to a JSON file
                json.dump(simulation_results, json_file, indent=4)
            print(f"  Progress saved to {results_filename}")
        except IOError as error:
            print(f"  Error saving JSON results: {error}")
        except TypeError as error:
            # This usually means some data in the results dict is not JSON-serializable
            print(f"  Error serializing results to JSON: {error}")


    # --- End of All Trial Sets ---
    total_end_time: float = time.time()
    total_duration: float = total_end_time - total_start_time
    print(f"\n--- All Trials Completed ({total_duration:.2f}s) ---")

    # Print final aggregate results to console
    print("\nFinal Aggregate Results:")
    pprint.pprint(simulation_results)

    # --- Save Final Summary Text File ---
    # Use the timestamp from the *last* saved JSON file
    if last_results_timestamp:
        summary_filename = f'results/summary/toptrumps_summary_{last_results_timestamp}.txt'
        try:
            # make parent folders for writing file, if needed
            dirpath = os.path.dirname(summary_filename)
            if dirpath:
                os.makedirs(dirpath, exist_ok=True)
            with open(summary_filename, 'w') as summary_file:
                # Write header information
                summary_file.write(f"Top Trumps Simulation Results ({time.strftime('%Y-%m-%d')})\n")
                summary_file.write(f"Total Duration: {total_duration:.2f} seconds\n")
                summary_file.write("="*50 + "\n")
                # Iterate through the collected results for each configuration
                for config_key, result_data in simulation_results.items():
                    summary_file.write(f"Configuration Key: {config_key}\n")
                    summary_file.write(f"  P1 ({result_data['player1_strategy']}) Wins: {result_data['player1_wins']}\n")
                    summary_file.write(f"  P2 ({result_data['player2_strategy']}) Wins: {result_data['player2_wins']}\n")
                    summary_file.write(f"  Draws:              {result_data['draws']}\n")
                    # Report errors if any occurred
                    if result_data['errors'] > 0:
                         summary_file.write(f"  Errors:             {result_data['errors']}\n")
                    # Verify total game count matches expected trials
                    total_recorded_games = result_data['player1_wins'] + result_data['player2_wins'] + result_data['draws'] + result_data['errors']
                    summary_file.write(f"  Total Games Run:    {total_recorded_games} / {result_data['number_trials']}\n")
                    summary_file.write(f"  Duration (set):     {result_data['duration_seconds']}s\n")
                    summary_file.write("-" * 30 + "\n")
            print(f"\nFinal summary saved to {summary_filename}")
        except IOError as error:
            print(f"\nError saving final text summary: {error}")
        except Exception as error:
            # Catch any other unexpected errors during summary writing
            print(f"\nUnexpected error saving final text summary: {error}")
    else:
        # This case handles if the trials_config list was empty and no results were saved
        print("\nCould not save text summary (no trial sets were run/results saved).")

    # Return the final results dictionary
    return simulation_results

# ==============================
# --- Example Single Game Runner ---
# ==============================

def run_single_game(player1_strategy_name: str = 'Goliath',
                    player2_strategy_name: str = 'EXPert',
                    number_of_cards: int = 30,
                    number_of_fields: int = 5,
                    player1_expert_m_value: float = 0.5, # Descriptive name
                    player2_expert_m_value: float = 0.5, # Descriptive name
                    player1_goliath_debug_flag: bool = False, # Descriptive name
                    player2_goliath_debug_flag: bool = False, # Descriptive name
                    engine_debug_flag: bool = False # Descriptive name
                    ):
    """Runs and prints detailed output for a single game."""
    print(f"\n--- Running Single Game: {player1_strategy_name} vs {player2_strategy_name} ---")
    print(f"Cards={number_of_cards}, Fields={number_of_fields}")

    # --- 1. Generate Deck Info ---
    try:
        # Request all return values, including the full reference deck
        p1_initial_deck_df, p2_initial_deck_df, field_names_list, full_deck_reference = \
            generate_decks(number_of_cards, number_of_fields)
        # Engine and agents need the full reference deck
        if full_deck_reference is None:
            raise ValueError("Full reference deck is required but was not generated.")
    except Exception as error:
        print(f"Error generating decks for single game: {error}")
        return # Cannot proceed without decks

    # --- 2. Create Agents ---
    try:
        # Get initial IDs as sets for agent initialization
        p1_initial_ids = set(p1_initial_deck_df['id'])
        p2_initial_ids = set(p2_initial_deck_df['id'])

        # Create Player 1 Agent using descriptive keywords
        player1 = create_agent(
            strategy=player1_strategy_name,
            my_initial_card_ids=p1_initial_ids,
            full_game_deck_ref_df=full_deck_reference,
            EXPert_m_value=player1_expert_m,
            goliath_debug_flag=player1_goliath_debug_flag if player1_strategy_name.lower()=='goliath' else False,
            number_of_fields=number_of_fields,
            number_of_cards=number_of_cards
            # Add randmax_fraction here if testing RandMaxer, e.g., randmax_fraction=0.7
        )
        # Create Player 2 Agent
        player2 = create_agent(
            strategy=player2_strategy_name,
            my_initial_card_ids=p2_initial_ids,
            full_game_deck_ref_df=full_deck_reference,
            EXPert_m_value=player2_expert_m,
            goliath_debug_flag=player2_goliath_debug_flag if player2_strategy_name.lower()=='goliath' else False,
            number_of_fields=number_of_fields,
            number_of_cards=number_of_cards
            # Add randmax_fraction here if needed
        )
    except Exception as error:
        print(f"Error creating agents for single game: {error}")
        return # Cannot proceed without agents

    # --- 3. Create Game Engine ---
    try:
        # Pass initial DFs, field names, and the essential full ref deck
        game_engine = GameEngine(
            player1_agent=player1,
            player2_agent=player2,
            initial_deck1_df=p1_initial_deck_df, # Engine extracts IDs from these
            initial_deck2_df=p2_initial_deck_df,
            field_names=field_names_list,
            full_game_ref_deck=full_deck_reference, # Engine needs ref deck
            engine_debug=engine_debug_flag
        )
    except Exception as error:
        print(f"Error creating GameEngine for single game: {error}")
        return # Cannot proceed without engine

    # --- 4. Run Game ---
    game_result_string = "Not Run"
    try:
        # Set print_tricks=True to see detailed turn-by-turn output for single game
        game_result_string = game_engine.play_game(print_tricks=True, max_tricks=2000) # Use a reasonable max_tricks
    except Exception as error:
        print(f"Error occurred during single game play: {error}")
        game_result_string = "Error during play"

    # --- Print Final Result ---
    print(f"\n--- Single Game Finished ---")
    if game_result_string in ['player1', 'player2', 'draw']:
        print(f"Result: {game_result_string.upper()}")
    else:
        # Print error message if the game didn't conclude normally
        print(f"Result: {game_result_string}")


# ==============================
# --- Original Trials Config Def ---
# ==============================
def define_original_trials_config() -> List[List[Any]]:
    """
    Defines the trial configuration list based on the original script's setup.
    Each inner list: [p1_strat, p2_strat, n_cards, n_trials, p1_rmax?, p2_rmax?, p1_expM?, p2_expM?]
    """
    # Define parameters used in the configurations
    config_num_cards = 50
    config_num_trials = 5000

    # List of trial configurations
    original_config = [
        # Main matchups
        ['rander',    'maxer',     config_num_cards, config_num_trials, None, None, None, None],
        ['maxer',     'rander',    config_num_cards, config_num_trials, None, None, None, None],
        ['maxer',     'meanermax', config_num_cards, config_num_trials, None, None, None, None],
        ['meanermax', 'maxer',     config_num_cards, config_num_trials, None, None, None, None],
        ['meanermax', 'EXPert',    config_num_cards, config_num_trials, None, None, None, 0.5],
        ['EXPert',    'meanermax', config_num_cards, config_num_trials, None, None, 0.5, None],
        ['EXPert',    'Goliath',   config_num_cards, config_num_trials, None, None, 0.5, 0.5],
        ['Goliath',   'EXPert',    config_num_cards, config_num_trials, None, None, 0.5, 0.5],
        ['maxer',     'EXPert',    config_num_cards, config_num_trials, None, None, None, 0.5],
        ['EXPert',    'maxer',     config_num_cards, config_num_trials, None, None, 0.5, None],
        ['maxer',     'Goliath',   config_num_cards, config_num_trials, None, None, None, 0.5],
        ['Goliath',   'maxer',     config_num_cards, config_num_trials, None, None, 0.5, None],
        ['meanermax', 'Goliath',   config_num_cards, config_num_trials, None, None, None, 0.5],
        ['Goliath',   'meanermax', config_num_cards, config_num_trials, None, None, 0.5, None],
        # Self-play baselines
        ['rander',    'rander',    config_num_cards, config_num_trials, None, None, None, None],
        ['maxer',     'maxer',     config_num_cards, config_num_trials, None, None, None, None],
        ['meanermax', 'meanermax', config_num_cards, config_num_trials, None, None, None, None],
        ['EXPert',    'EXPert',    config_num_cards, config_num_trials, None, None, 0.5, 0.5],
        ['Goliath',   'Goliath',   config_num_cards, config_num_trials, None, None, 0.5, 0.5],
    ]
    return original_config

# ==============================
# --- Main Execution Block ---
# ==============================
if __name__ == "__main__":
    # Set seeds at the start for reproducibility across the entire run (optional)
    random.seed(123)
    np.random.seed(123)

    # === CHOOSE MODE TO RUN ===

    # --- Option 1: Run a Single Detailed Game ---
    # Useful for debugging a specific agent or watching interactions.
    # run_single_game(
    #      player1_strategy_name='Goliath',   # Example: 'user', 'maxer', 'EXPert', etc.
    #      player2_strategy_name='EXPert',
    #      number_of_cards=20,                # Smaller deck for faster game
    #      number_of_fields=5,
    #      player1_expert_m=0.5,              # M-value if P1 is EXPert/Goliath
    #      player2_expert_m=0.5,              # M-value if P2 is EXPert/Goliath
    #      player1_goliath_debug_flag=False,  # Set True to see P1 Goliath logs
    #      player2_goliath_debug_flag=False,  # Set True to see P2 Goliath logs
    #      engine_debug_flag=False            # Set True for Game Engine logs
    # )

    # --- Option 2: Run Batch Trials ---
    # Use this for performance comparisons between agents over many games.
    print("\n" + "="*60 + "\nStarting Batch Trials...\n" + "="*60 + "\n")
    list_of_trial_configs = define_original_trials_config()
    # Remember to adjust num_trials inside define_original_trials_config for speed.
    batch_results = run_trials(
         trials_config=list_of_trial_configs,
         number_of_fields=5, # Standard number of fields for these trials
         max_tricks_per_game=100000, # Allow potentially long games
         # Keep debug flags False for batch runs to avoid excessive console output
         goliath_player1_debug=False,
         goliath_player2_debug=False,
         engine_debug_flag=False
    )
    print("\nBatch run complete. Results saved to .json and .txt files.")


    print("\nScript finished.")