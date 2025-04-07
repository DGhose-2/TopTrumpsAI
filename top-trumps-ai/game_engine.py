# game_engine.py
"""
Defines the GameEngine class that manages the Top Trumps game flow.
Uses efficient deques internally for deck management and provides
information to agents strictly based on allowed constraints (API).
"""
import random
from typing import Optional, Tuple, Dict, Any, Set, List, Union, Deque # Ensure all needed types are imported
from collections import deque # Use deque for efficient deck operations
import numpy as np
import pandas as pd
from agents import Agent # Import base Agent class for type hinting

class GameEngine:
    """
    Manages the state and logic of a Top Trumps game between two agents.

    Uses deques of card IDs internally for efficient deck manipulation.
    Provides information to agents strictly through the defined Agent API
    (choose_field, update_state methods). Requires the full reference deck
    at initialization to look up card stats when needed for the API.
    """
    def __init__(self,
                 player1_agent: Agent,
                 player2_agent: Agent,
                 initial_deck1_df: pd.DataFrame, # Initial DF needed to get starting IDs
                 initial_deck2_df: pd.DataFrame, # Initial DF needed to get starting IDs
                 field_names: List[str],           # List of valid field names
                 full_game_ref_deck: pd.DataFrame, # Required for stat lookups
                 engine_debug: bool = False):      # Flag for engine debug prints
        """
        Initializes the game engine.

        Args:
            player1_agent: An initialized Agent object for Player 1.
            player2_agent: An initialized Agent object for Player 2.
            initial_deck1_df: DataFrame for Player 1's starting cards (used to get IDs).
            initial_deck2_df: DataFrame for Player 2's starting cards (used to get IDs).
            field_names: List of valid field column names (strings).
            full_game_ref_deck: DataFrame containing all cards and stats in the game.
            engine_debug: If True, enable engine-specific debug prints.

        Raises:
            TypeError: If agents are not Agent instances.
            ValueError: If required inputs (field_names, ref deck, deck columns) are invalid.
        """
        # --- Input Validation ---
        if not isinstance(player1_agent, Agent) or not isinstance(player2_agent, Agent):
            raise TypeError("player1_agent and player2_agent must be instances of Agent")
        if not isinstance(field_names, list) or not field_names:
            raise ValueError("field_names must be a non-empty list")
        if full_game_ref_deck is None or not isinstance(full_game_ref_deck, pd.DataFrame) or 'id' not in full_game_ref_deck.columns:
             raise ValueError("A valid full_game_ref_deck DataFrame with an 'id' column is required.")
        if 'id' not in initial_deck1_df.columns or 'id' not in initial_deck2_df.columns:
            raise ValueError("Initial decks must contain an 'id' column to extract IDs.")

        # --- Store Agents and Config ---
        self.player1_agent: Agent = player1_agent
        self.player2_agent: Agent = player2_agent
        self.field_colnames: List[str] = list(field_names) # Ensure it's a list copy
        self._debug: bool = engine_debug

        # --- Store initial IDs in deques for efficient reset ---
        self._initial_deck1_ids: deque[str] = deque(initial_deck1_df['id'])
        self._initial_deck2_ids: deque[str] = deque(initial_deck2_df['id'])
        self.n_cards_total: int = len(self._initial_deck1_ids) + len(self._initial_deck2_ids)

        # --- Store central reference deck, indexed by ID for fast lookups ---
        try:
            # Create a copy before setting index to avoid modifying original passed df
            ref_deck_copy = full_game_ref_deck.copy()
            # Ensure 'id' column exists before setting index
            if 'id' not in ref_deck_copy.columns:
                 raise ValueError("Cannot set index: 'id' column missing in reference deck.")
            # Set 'id' as index, but keep 'id' also as a regular column for compatibility
            ref_deck_copy.set_index('id', inplace=True, drop=False)
            self.full_game_deck_ref_indexed: pd.DataFrame = ref_deck_copy
        except KeyError:
             raise ValueError("Failed to set 'id' as index on reference deck. Does the column exist?")
        except Exception as e:
             raise RuntimeError(f"Failed to prepare indexed reference deck: {e}")


        # Verify all initial card IDs exist in the reference deck's index
        all_ref_ids: Set[str] = set(self.full_game_deck_ref_indexed.index)
        p1_missing_ids = set(self._initial_deck1_ids) - all_ref_ids
        p2_missing_ids = set(self._initial_deck2_ids) - all_ref_ids
        if p1_missing_ids:
            raise ValueError(f"Player 1's initial deck contains IDs not found in the reference deck: {p1_missing_ids}")
        if p2_missing_ids:
            raise ValueError(f"Player 2's initial deck contains IDs not found in the reference deck: {p2_missing_ids}")
        # Verify total card count matches reference deck size
        if self.n_cards_total != len(all_ref_ids):
             print(f"WARN Engine Init: Total initial cards ({self.n_cards_total}) != reference deck size ({len(all_ref_ids)}).")

        # --- Game State Variables (using deques for decks) ---
        self.player1_deck_ids: deque[str] = deque()
        self.player2_deck_ids: deque[str] = deque()
        self.draw_pile_ids: deque[str] = deque() # Stores IDs only
        self.trick_starter_index: int = 0 # Player 1 starts (index 0)
        self.last_trick_outcome_for_p1: Optional[str] = None # Tracks last outcome for turn logic
        self.trick_count: int = 0 # Counter for tricks played
        self._min_compare_val: int = -999999 # Value used if card stat is missing/invalid


    def setup_game(self):
        """Initializes or resets the game state for a new game using deques."""
        self._log_engine("Setting up new game...")
        # Reset decks from the stored initial ID deques
        self.player1_deck_ids = self._initial_deck1_ids.copy()
        self.player2_deck_ids = self._initial_deck2_ids.copy()
        # Reset draw pile
        self.draw_pile_ids.clear()
        # Reset turn state
        self.trick_starter_index = 0 # Player 1 starts the first trick
        self.last_trick_outcome_for_p1 = None # No outcome before the first trick
        self.trick_count = 0
        self._log_engine(f"Game setup complete. P1 cards: {len(self.player1_deck_ids)}, P2 cards: {len(self.player2_deck_ids)}")


    def _log_engine(self, *args):
        """Prints engine debug messages if debug flag is True."""
        if self._debug:
            # Include trick count for context
            print(f"ENGINE (T{self.trick_count}):", *args)


    def _get_card_df(self, card_id: str) -> pd.DataFrame:
        """
        Looks up a single card's data from the indexed reference deck.
        Returns a 1-row DataFrame matching the structure agents expect.
        """
        try:
            # Use .loc with list of one ID to ensure DataFrame return type
            card_data_df = self.full_game_deck_ref_indexed.loc[[card_id]]
            return card_data_df
        except KeyError:
            # This indicates a serious issue - an ID exists in a deck deque but not the ref index
            print(f"CRITICAL ENGINE ERROR: Card ID '{card_id}' not found in reference index during lookup!")
            # Return an empty DataFrame with correct columns to potentially allow graceful failure
            return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)
        except Exception as error:
            print(f"CRITICAL ENGINE ERROR: Failed DataFrame lookup for card ID '{card_id}': {error}")
            return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)


    def _get_cards_df(self, card_ids: Union[List[str], Set[str], Deque[str]]) -> pd.DataFrame:
         """
         Looks up multiple cards' data from the indexed reference deck.
         Used to create DataFrames for agent update_state calls (won cards, pot).
         Returns DataFrame with stats for the found cards.
         """
         # Handle empty input list/set/deque
         if not card_ids:
             return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)

         # Ensure input is a list for .loc indexing
         ids_list = list(card_ids)

         # Pre-check which requested IDs actually exist in the reference index
         all_reference_ids = self.full_game_deck_ref_indexed.index
         valid_ids_list = [card_id for card_id in ids_list if card_id in all_reference_ids]

         # Log a warning if some requested IDs were not found
         missing_ids = set(ids_list) - set(valid_ids_list)
         if missing_ids:
              print(f"WARN ENGINE: Card IDs not found in reference index during multi-lookup: {missing_ids}")

         # If no valid IDs remain after filtering, return empty DataFrame
         if not valid_ids_list:
             return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)

         # Perform lookup using the filtered list of valid IDs
         try:
              cards_data_df = self.full_game_deck_ref_indexed.loc[valid_ids_list]
              return cards_data_df
         except KeyError as error:
             # This might occur if there's a race condition or unexpected index issue
             print(f"CRITICAL ENGINE ERROR: KeyError during multi-lookup for valid IDs {valid_ids_list}: {error}")
             return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)
         except Exception as error:
             # Catch other potential lookup errors
             print(f"CRITICAL ENGINE ERROR: Failed multi-lookup for IDs {valid_ids_list}: {type(error).__name__} - {error}")
             return pd.DataFrame(columns=self.full_game_deck_ref_indexed.columns)


    def _get_game_state_for_agent(self, agent_index: int) -> Optional[Dict[str, Any]]:
        """
        Prepares the state view dictionary for an agent's `choose_field` call,
        adhering strictly to information constraints. Uses deques internally.
        """
        # Determine which player is 'my' and 'opponent' from agent_index perspective
        my_deck_ids = self.player1_deck_ids if agent_index == 0 else self.player2_deck_ids
        opponent_deck_ids = self.player2_deck_ids if agent_index == 0 else self.player1_deck_ids

        # If the agent's deck is empty, they cannot choose a field
        if not my_deck_ids:
            self._log_engine(f"Agent {agent_index+1} has no cards.")
            return None # Signal agent cannot play

        # Get the ID of the agent's top card (without removing it yet)
        top_card_id = my_deck_ids[0]
        # Look up the stats for this top card to create the DataFrame for the agent
        top_card_df = self._get_card_df(top_card_id)

        # Prepare the dictionary containing ONLY the allowed information
        agent_state_view = {
            "my_top_card_df": top_card_df, # Agent sees their top card stats
            "opponent_hand_size": len(opponent_deck_ids), # Agent knows opponent deck size
            "available_fields": self.field_colnames,   # Agent knows valid fields
            # NOTE: Agent does NOT receive its full current deck ID set here. It must track internally.
        }
        return agent_state_view


    def play_trick(self, print_trick_details: bool = True) -> Optional[str]:
        """
        Plays a single trick/turn of the game using efficient deque operations.

        1. Determines starting player.
        2. Gets state for starter, calls agent's choose_field.
        3. Retrieves card IDs and looks up stats.
        4. Compares values and determines outcome.
        5. Updates player deck ID deques according to rules.
        6. Prepares info DFs (played cards, won cards, pot) for update_state.
        7. Notifies both agents via update_state.
        8. Updates next starter and checks for game over.

        Args:
            print_trick_details: If True, prints trick info to console.

        Returns:
            'player1' or 'player2' if game ended, otherwise None.
        """
        self.trick_count += 1
        self._log_engine(f"--- Starting Trick {self.trick_count} ---")

        # Check for game over *before* the trick starts
        game_over_status = self.is_game_over()
        if game_over_status:
            self._log_engine(f"Game already over before trick start ({game_over_status} won).")
            self.trick_count -= 1 # Decrement count as trick didn't happen
            return game_over_status

        # Determine starting player based on last outcome (or P1 for first trick)
        # trick_starter_index is updated *after* each trick resolution
        current_starter_index = self.trick_starter_index
        current_agent = self.player1_agent if current_starter_index == 0 else self.player2_agent
        starter_name_log = f"P{current_starter_index+1}({current_agent.strategy_name})"
        self._log_engine(f"Starter: {starter_name_log}")

        # Get the limited game state view for the starting agent
        agent_state_view = self._get_game_state_for_agent(current_starter_index)
        if agent_state_view is None:
            # This case should ideally be caught by is_game_over, but check defensively
            self._log_engine(f"Starter {starter_name_log} has no cards, game should have ended.")
            self.trick_count -= 1
            return self.is_game_over() # Re-check game state

        # --- Agent Chooses Field ---
        chosen_field_name = None
        try:
            # Call agent's method with only the allowed arguments
            chosen_field_name = current_agent.choose_field(**agent_state_view)

            # Validate the agent's choice
            if chosen_field_name not in self.field_colnames:
                print(f"Warning: Agent {current_agent.strategy_name} chose invalid field '{chosen_field_name}'. Choosing random.")
                chosen_field_name = random.choice(self.field_colnames) if self.field_colnames else "Field0" # Fallback
        except Exception as error:
            print(f"CRITICAL Error during agent {current_agent.strategy_name} choose_field: {type(error).__name__} - {error}. Choosing random field.")
            chosen_field_name = random.choice(self.field_colnames) if self.field_colnames else "Field0" # Fallback

        self._log_engine(f"{starter_name_log} chose field: {chosen_field_name}")

        # --- Get Cards and Values ---
        # Ensure both players still have cards before proceeding
        if not self.player1_deck_ids or not self.player2_deck_ids:
             self._log_engine("One player deck empty after field choice? Game should end.")
             self.trick_count -= 1
             return self.is_game_over()

        # Get the IDs of the cards being played
        player1_card_id = self.player1_deck_ids[0]
        player2_card_id = self.player2_deck_ids[0]
        self._log_engine(f"P1 plays '{player1_card_id}', P2 plays '{player2_card_id}'")

        # Look up the full data for these cards (needed for value comparison and update_state)
        player1_card_played_df = self._get_card_df(player1_card_id)
        player2_card_played_df = self._get_card_df(player2_card_id)

        # Get the actual values for comparison
        player1_value = self._get_card_value(player1_card_played_df, chosen_field_name)
        player2_value = self._get_card_value(player2_card_played_df, chosen_field_name)
        self._log_engine(f"Comparing on '{chosen_field_name}': P1 Value={player1_value}, P2 Value={player2_value}")

        # --- Print Trick Info if Requested ---
        if print_trick_details:
            print("-" * 30)
            print(f"Trick {self.trick_count}: {starter_name_log} chooses {chosen_field_name}")
            print(f"  P1 ({self.player1_agent.strategy_name}) Card: {player1_card_id} ({chosen_field_name}={player1_value})")
            print(f"  P2 ({self.player2_agent.strategy_name}) Card: {player2_card_id} ({chosen_field_name}={player2_value})")

        # --- Determine Winner and Update Decks (using deques) ---
        # Remove played cards from the top of deques FIRST
        played_id_p1 = self.player1_deck_ids.popleft()
        played_id_p2 = self.player2_deck_ids.popleft()

        # Initialize tracking variables for the trick outcome
        trick_outcome_for_p1: str = 'draw' # Default outcome
        won_by_p1_card_ids: List[str] = [] # IDs opponent lost + pot gained by P1
        won_by_p2_card_ids: List[str] = [] # IDs P1 lost + pot gained by P2
        pot_card_ids_for_update: List[str] = [] # IDs involved in a draw

        # Compare values to determine winner
        if player1_value > player2_value:
            # --- P1 Wins Trick ---
            trick_outcome_for_p1 = 'win'
            self._log_engine("Outcome: P1 Wins Trick")
            if print_trick_details: print("  Result: P1 wins trick.")
            # Collect IDs P1 won: opponent's card + any cards from draw pile
            won_by_p1_card_ids = [played_id_p2] + list(self.draw_pile_ids)
            # P1 gets own card back first, then the winnings
            cards_to_add_to_p1_deck = [played_id_p1] + won_by_p1_card_ids
            self.player1_deck_ids.extend(cards_to_add_to_p1_deck)
            # Clear the draw pile
            if self.draw_pile_ids: self._log_engine(f" P1 also wins draw pile ({len(self.draw_pile_ids)} cards)")
            self.draw_pile_ids.clear()

        elif player2_value > player1_value:
            # --- P2 Wins Trick ---
            trick_outcome_for_p1 = 'loss'
            self._log_engine("Outcome: P2 Wins Trick")
            if print_trick_details: print("  Result: P2 wins trick.")
            # Collect IDs P2 won: P1's card + any cards from draw pile
            won_by_p2_card_ids = [played_id_p1] + list(self.draw_pile_ids)
            # P2 gets own card back first, then the winnings
            cards_to_add_to_p2_deck = [played_id_p2] + won_by_p2_card_ids
            self.player2_deck_ids.extend(cards_to_add_to_p2_deck)
            # Clear the draw pile
            if self.draw_pile_ids: self._log_engine(f" P2 also wins draw pile ({len(self.draw_pile_ids)} cards)")
            self.draw_pile_ids.clear()

        else:
            # --- Draw ---
            trick_outcome_for_p1 = 'draw'
            self._log_engine("Outcome: Draw")
            if print_trick_details: print("  Result: Draw.")
            # Add played cards to the draw pile. Order depends on starter.
            # Convention: Loser's card (if applicable), then Starter's card.
            # For draw, non-starter's card then starter's card.
            if current_starter_index == 0: # P1 started
                 cards_to_add_to_draw_pile = [played_id_p2, played_id_p1]
            else: # P2 started
                 cards_to_add_to_draw_pile = [played_id_p1, played_id_p2]
            # Add cards to draw pile deque
            self.draw_pile_ids.extend(cards_to_add_to_draw_pile)
            pot_card_ids_for_update = cards_to_add_to_draw_pile # Store for update_state
            self._log_engine(f" Added {len(cards_to_add_to_draw_pile)} cards to draw pile. Total pile size: {len(self.draw_pile_ids)}")

        # --- Prepare Agent Notification Data ---
        # Create the necessary DataFrames from the collected IDs just before calling update_state
        p1_won_cards_df = self._get_cards_df(won_by_p1_card_ids)
        p2_won_cards_df = self._get_cards_df(won_by_p2_card_ids)
        pot_cards_for_update_df = self._get_cards_df(pot_card_ids_for_update)

        # Dictionary for Player 1's update
        p1_update_info = {
            "outcome": trick_outcome_for_p1,
            "chosen_field": chosen_field_name,
            "my_card_played_df": player1_card_played_df, # The card P1 played
            # Opponent's ID is revealed only on win or draw
            "opponent_actual_card_id": player2_card_id if trick_outcome_for_p1 in ['win', 'draw'] else None,
            "opponent_revealed_value": player2_value, # Opponent's value is always revealed
            "pot_cards_df": pot_cards_for_update_df, # Cards involved in a draw
            "cards_i_won_df": p1_won_cards_df, # Cards P1 gained (P2's card + pot)
            "cards_opponent_won_df": p2_won_cards_df # Cards P2 gained (P1's card + pot)
        }

        # Determine outcome from Player 2's perspective
        p2_outcome = {'win': 'loss', 'loss': 'win', 'draw': 'draw'}.get(trick_outcome_for_p1)
        # Dictionary for Player 2's update
        p2_update_info = {
            "outcome": p2_outcome,
            "chosen_field": chosen_field_name,
            "my_card_played_df": player2_card_played_df, # The card P2 played
            # Opponent's ID (P1) is revealed only if P2 won or drew
            "opponent_actual_card_id": player1_card_id if p2_outcome in ['win', 'draw'] else None,
            "opponent_revealed_value": player1_value, # P1's value is always revealed
            "pot_cards_df": pot_cards_for_update_df,
            "cards_i_won_df": p2_won_cards_df, # Cards P2 gained (P1's card + pot)
            "cards_opponent_won_df": p1_won_cards_df # Cards P1 gained (P2's card + pot)
        }

        # --- Notify Agents ---
        self._log_engine("Notifying agents of outcome...")
        try:
            self.player1_agent.update_state(**p1_update_info)
        except Exception as error:
             # Log errors during agent updates, but continue the game
             print(f"CRITICAL ENGINE ERROR calling P1 ({self.player1_agent.strategy_name}) update_state: {type(error).__name__} - {error}")
        try:
            self.player2_agent.update_state(**p2_update_info)
        except Exception as error:
             print(f"CRITICAL ENGINE ERROR calling P2 ({self.player2_agent.strategy_name}) update_state: {type(error).__name__} - {error}")

        # --- Update Starter for Next Trick ---
        # Winner starts next trick. If draw, the *same* player starts again.
        self.last_trick_outcome_for_p1 = trick_outcome_for_p1
        if trick_outcome_for_p1 == 'win':
            self.trick_starter_index = 0 # P1 starts next
        elif trick_outcome_for_p1 == 'loss':
            self.trick_starter_index = 1 # P2 starts next
        # If 'draw', self.trick_starter_index remains unchanged (previous starter continues)

        # --- Log end of trick state ---
        self._log_engine(f"Trick end. P1 cards: {len(self.player1_deck_ids)}, P2 cards: {len(self.player2_deck_ids)}, Draw pile: {len(self.draw_pile_ids)}")
        self._log_engine(f"Next starter index: {self.trick_starter_index}")

        # --- Check for Game Over After Trick ---
        return self.is_game_over()


    def _get_card_value(self, card_df: pd.DataFrame, field: str) -> int:
        """
        Safely retrieves and converts a card's value from its DataFrame for comparison.
        Uses the minimum comparison value if the value is missing or invalid.
        """
        # Check if DataFrame is valid and field exists
        if card_df.empty or field not in card_df.columns:
            return self._min_compare_val
        try:
            # Get the value, handling potential single-element Series/scalar
            value = card_df[field].iloc[0]
            # Convert to standard Python int if not NA, else return min value
            if pd.notna(value):
                return int(value) # Convert Int64/float to int
            else:
                # Log if value is NA
                card_id = card_df['id'].iloc[0] if 'id' in card_df.columns else 'Unknown ID'
                self._log_engine(f"Warning: Card '{card_id}' has missing value (NA) for field '{field}'. Using {self._min_compare_val}.")
                return self._min_compare_val
        except (IndexError, ValueError, TypeError) as error:
            # Catch potential errors during access or conversion
            card_id = card_df['id'].iloc[0] if 'id' in card_df.columns else 'Unknown ID'
            print(f"ENGINE ERROR converting value for card '{card_id}', field '{field}': {error}. Using {self._min_compare_val}.")
            return self._min_compare_val


    def is_game_over(self) -> Optional[str]:
        """
        Checks if the game has ended (one player has no cards in their deque).

        Returns:
            'player1': If Player 1 wins (Player 2 has 0 cards).
            'player2': If Player 2 wins (Player 1 has 0 cards).
            None: If the game is ongoing.
        """
        player1_card_count = len(self.player1_deck_ids)
        player2_card_count = len(self.player2_deck_ids)

        # --- Sanity Check: Card Conservation ---
        # Ensure total cards in play (both decks + draw pile) equals initial total.
        total_cards_in_play = player1_card_count + player2_card_count + len(self.draw_pile_ids)
        if total_cards_in_play != self.n_cards_total:
             # This indicates a bug where cards are lost or duplicated.
             self._log_engine(f"CRITICAL WARNING: Card count mismatch! Expected {self.n_cards_total}, "
                              f"Found {total_cards_in_play} (P1:{player1_card_count}, P2:{player2_card_count}, Draw:{len(self.draw_pile_ids)})")
             # Depending on severity, could raise an error or try to continue.

        # --- Standard Win Conditions ---
        # Check if Player 1 ran out of cards.
        if player1_card_count == 0:
            self._log_engine("Game Over: Player 1 has 0 cards. Player 2 wins.")
            return 'player2'
        # Check if Player 2 ran out of cards.
        if player2_card_count == 0:
            self._log_engine("Game Over: Player 2 has 0 cards. Player 1 wins.")
            return 'player1'

        # If neither player has 0 cards, the game continues.
        return None


    def play_game(self, print_tricks: bool = False, max_tricks: int = 5000) -> str:
        """
        Plays a complete game from setup until a winner is found or max_tricks reached.

        Args:
            print_tricks: Whether to print details of each trick.
            max_tricks: Maximum number of tricks before ending by card count.

        Returns:
            'player1', 'player2', or 'draw' indicating the game result.
        """
        self.setup_game() # Initialize/reset game state using deques

        # Main game loop
        while self.trick_count < max_tricks:
            # Play one trick and check if it ended the game
            winner = self.play_trick(print_trick_details=print_tricks)
            if winner:
                # Game ended naturally by a player running out of cards
                if print_tricks: # Print final outcome clearly if details were shown
                    print("="*30)
                    print(f"GAME OVER after {self.trick_count} tricks.")
                    final_p1_count = len(self.player1_deck_ids)
                    final_p2_count = len(self.player2_deck_ids)
                    print(f"Winner: {winner.upper()}")
                    print(f"Final Card Counts: P1={final_p1_count}, P2={final_p2_count}")
                    print("="*30)
                return winner # Return 'player1' or 'player2'

            # Optional: Add more sophisticated loop/stalemate detection here if needed

        # --- Max Tricks Reached ---
        # Game did not conclude naturally within the limit.
        print(f"\nWarning: Game reached max tricks ({max_tricks}). Determining winner by card count.")
        final_p1_count = len(self.player1_deck_ids)
        final_p2_count = len(self.player2_deck_ids)
        final_draw_count = len(self.draw_pile_ids)
        print(f"  Final Card Counts: P1={final_p1_count}, P2={final_p2_count}, Draw Pile={final_draw_count}")

        # Determine winner based on who has more cards
        if final_p1_count > final_p2_count:
            print("  Result: Player 1 wins by card count.")
            return 'player1'
        elif final_p2_count > final_p1_count:
            print("  Result: Player 2 wins by card count.")
            return 'player2'
        else:
            # Card counts are equal, declare a draw
            print("  Result: Draw (equal card counts at max tricks).")
            return 'draw'