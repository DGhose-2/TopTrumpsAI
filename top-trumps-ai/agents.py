# agents.py
"""
Defines the base Agent class and various AI agent implementations for Top Trumps.

Includes internal optimizations (indexed lookups) and reverted internal logic
(using DataFrame slicing) for Goliath and EXPert agents for better readability.

**Agent Information Constraints (Strict):**
Agents receive information ONLY through:
1. Initialization (`__init__`):
    - `full_game_deck_ref_df` (if needed for stats).
    - `my_initial_card_ids: Set[str]` (to know own starting cards).
2. `choose_field` arguments: `my_top_card_df`, `opponent_hand_size`,
   `available_fields`. (Agent CANNOT see its full current deck here directly).
3. `update_state` arguments: Details about the completed trick outcome.

Agents MUST track their own current deck IDs and any opponent state internally,
updating based on init info and `update_state` calls.
"""

import random
import time
import pprint
from typing import Optional, Any, List, Set, Union, Dict, Deque # Added Deque hint
from collections import deque # Imported for Goliath _propagate helper

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # Used by MeanerMaxAgent

# ==============================
# --- Agent Base Class ---
# ==============================
class Agent:
    """
    Base class for all Top Trumps agents.

    Handles basic initialization, stores reference data, and implements
    tracking of the agent's own current set of card IDs via the update_state method.
    Subclasses must implement the choose_field method and can override update_state
    (remembering to call super().update_state() or reimplement own card tracking).
    """
    def __init__(self,
                 strategy_name: str,
                 my_initial_card_ids: Set[str],
                 full_game_deck_ref_df: Optional[pd.DataFrame] # Needed by some subclasses
                ):
        """
        Initializes the base agent.

        Args:
            strategy_name: Name of the agent's strategy (e.g., 'maxer', 'Goliath').
            my_initial_card_ids: Set of string IDs the agent starts the game with.
            full_game_deck_ref_df: Reference DataFrame of all cards in the game.
                                     Can be None if the agent doesn't need it.
        """
        if not isinstance(strategy_name, str) or not strategy_name:
            raise ValueError("Agent requires a non-empty strategy_name.")
        if my_initial_card_ids is None: # Factory should always provide this
             raise ValueError("Agent requires my_initial_card_ids at initialization.")

        self.strategy_name: str = strategy_name

        # --- Own Deck Tracking ---
        # Store the initial set and track the current set throughout the game.
        self.my_current_card_ids: Set[str] = my_initial_card_ids.copy()

        # --- Store Reference Deck Info (if provided) ---
        # Store a copy to prevent external modifications.
        self.full_game_deck_ref_df: Optional[pd.DataFrame] = full_game_deck_ref_df.copy() if full_game_deck_ref_df is not None else None
        # Store an indexed version for potentially faster lookups by subclasses
        self._indexed_ref_df: Optional[pd.DataFrame] = None
        # Store field names for convenience
        self.field_colnames: List[str] = []

        # If a reference deck was provided, validate and index it
        if self.full_game_deck_ref_df is not None:
            self._validate_and_index_ref_deck()

    def _validate_and_index_ref_deck(self):
        """Validates the reference deck and creates an indexed version."""
        # Check essential columns
        if 'id' not in self.full_game_deck_ref_df.columns:
            raise ValueError("Reference deck must contain an 'id' column.")
        self.field_colnames = [col for col in self.full_game_deck_ref_df.columns if col.startswith('Field')]
        if not self.field_colnames:
            raise ValueError("Reference deck must contain 'FieldX' columns.")

        # Ensure numeric types (Int64) in the stored copy before indexing
        safe_fill_value: int = -1 # Placeholder for potential NaNs during conversion
        try:
            for col in self.field_colnames:
                 numeric_col = pd.to_numeric(self.full_game_deck_ref_df[col], errors='coerce')
                 self.full_game_deck_ref_df[col] = numeric_col.fillna(safe_fill_value).round().astype('Int64')
        except Exception as error:
             raise TypeError(f"Agent '{self.strategy_name}' failed ref deck Int64 conversion: {error}")

        # Create indexed copy (used by optimized agents)
        try:
            # Index a fresh copy to leave self.full_game_deck_ref_df non-indexed
            indexed_copy = self.full_game_deck_ref_df.copy()
            indexed_copy.set_index('id', inplace=True, drop=False) # Keep 'id' column
            self._indexed_ref_df = indexed_copy
        except KeyError:
             raise ValueError("Failed to set 'id' index on reference deck (KeyError).")
        except Exception as error:
             print(f"WARN: Agent '{self.strategy_name}' failed to create indexed reference deck: {error}")
             self._indexed_ref_df = None # Ensure it's None if indexing failed


    def choose_field(self,
                     my_top_card_df: pd.DataFrame,
                     opponent_hand_size: int,
                     available_fields: List[str]
                     # Agent must use self.my_current_card_ids internally if needed
                     ) -> str:
        """
        Abstract method for choosing a field to compete on.

        Args:
            my_top_card_df: DataFrame containing only the agent's top card (1 row).
            opponent_hand_size: The number of cards the opponent currently holds.
            available_fields: A list of valid field names (strings) to choose from.

        Returns:
            The name (string) of the chosen field from `available_fields`.
        """
        raise NotImplementedError("Agent subclasses must implement choose_field")

    def update_state(self,
                     outcome: str, # 'win', 'loss', 'draw' for this agent
                     chosen_field: str, # Field competed on
                     my_card_played_df: pd.DataFrame, # DF of card this agent played
                     opponent_actual_card_id: Optional[str], # Opponent's ID (if win/draw), None if loss
                     opponent_revealed_value: Any, # Opponent card's value in chosen_field
                     pot_cards_df: pd.DataFrame, # Cards added to pot in a draw (both played cards)
                     cards_i_won_df: pd.DataFrame, # Cards this agent gained (opp card + pot)
                     cards_opponent_won_df: pd.DataFrame # Cards opponent gained (our played card + pot)
                    ):
        """
        Updates the agent's internal state after a trick resolves.
        **Crucially, tracks the agent's own set of card IDs.**
        Subclasses MUST call `super().update_state(...)` if they override this method
        to ensure their own card tracking remains correct, unless they reimplement
        this logic themselves.
        """
        # --- Base implementation updates OWN card set ---
        my_played_card_id: Optional[str] = None
        # Safely get the ID of the card played by this agent
        if not my_card_played_df.empty and 'id' in my_card_played_df.columns:
             try:
                 my_played_card_id = my_card_played_df['id'].iloc[0]
             except IndexError:
                  # This case should ideally not be reachable if input is always 1 row
                  print(f"WARN Base Update [{self.strategy_name}]: Could not get ID from my_card_played_df.")

        # 1. Remove card played by self from the tracked set
        if my_played_card_id:
            if my_played_card_id in self.my_current_card_ids:
                 self.my_current_card_ids.discard(my_played_card_id)
            # else: # Optional Warning for debugging state inconsistencies
            #     print(f"WARN Base Update [{self.strategy_name}]: Played card '{my_played_card_id}' was not in tracked set {self.my_current_card_ids}")


        # 2. Add cards won by self (opponent's card + any cards from pot)
        if not cards_i_won_df.empty and 'id' in cards_i_won_df.columns:
            won_ids: Set[str] = set(cards_i_won_df['id'])
            # Optional check: Ensure won cards weren't somehow already tracked
            # already_present = won_ids.intersection(self.my_current_card_ids)
            # if already_present:
            #     print(f"WARN Base Update [{self.strategy_name}]: Won cards {already_present} were already in tracked set?")
            self.my_current_card_ids.update(won_ids)

        # Note on 'loss'/'draw': If the agent lost or drew, cards_i_won_df is empty, so nothing is added.
        # The card the agent played (my_played_card_id) is correctly removed above regardless of outcome.
        # If it was a draw, the played card effectively leaves the hand for the pot.
        # If it was a loss, the played card leaves the hand and goes to the opponent (tracked by cards_opponent_won_df).

        # Subclasses can add their specific opponent tracking logic AFTER calling super().update_state(...)

# ==============================
# --- Standard Agent Implementations ---
# ==============================

class UserAgent(Agent):
    """Agent that prompts a human user for input via the console."""
    def __init__(self, my_initial_card_ids: Set[str], full_game_deck_ref_df: Optional[pd.DataFrame]):
        # Initialize base class (stores initial IDs, ref deck)
        super().__init__('user', my_initial_card_ids, full_game_deck_ref_df)

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """Displays card info and prompts the user for a field choice."""
        print("\n" + "="*30)
        print("Your turn!")
        print("Your top card:")
        try:
            card_id = my_top_card_df['id'].iloc[0]
            print(f"  Card ID: {card_id}")
        except (IndexError, KeyError):
            print("  Card ID: Error retrieving")

        # Display stats for the available fields
        print("  Stats:")
        if not my_top_card_df.empty:
            field_data = my_top_card_df.iloc[0] # Get the Series for the top card
            for index, field_name in enumerate(available_fields):
                value = field_data.get(field_name, 'N/A') # Safely get value
                print(f"    {index}: {field_name} = {value}")
        else:
            print("    Error retrieving stats.")

        # Display current game state info
        print(f"\nYour current deck size: {len(self.my_current_card_ids)}") # Uses internal tracked state
        print(f"Opponent has {opponent_hand_size} cards.")

        # Get valid user input for field choice
        field_indices = list(range(len(available_fields)))
        prompt = f"Choose field index ({', '.join(map(str, field_indices))}): "
        while True:
            try:
                field_index_str = input(prompt)
                field_index = int(field_index_str)
                # Validate user choice against available indices
                if field_index in field_indices:
                    chosen_field_name = available_fields[field_index]
                    print(f"--> You chose index {field_index} ({chosen_field_name})")
                    print("="*30)
                    return chosen_field_name
                else:
                    # Provide feedback if index is out of range
                    print(f"    Invalid index. Please choose from: {', '.join(map(str, field_indices))}")
            except ValueError:
                # Handle non-integer input
                print("    Invalid input. Please enter a number.")
            except EOFError:
                # Handle unexpected end of input (e.g., piped input ends)
                print("\nEOF detected. Choosing random field.")
                # Provide a fallback choice if input stream closes
                return random.choice(available_fields) if available_fields else "Field0"

    # update_state for tracking own cards is handled by the base Agent class

class MaxerAgent(Agent):
    """Agent that always chooses the field where its top card has the maximum raw value."""
    def __init__(self, my_initial_card_ids: Set[str], full_game_deck_ref_df: Optional[pd.DataFrame]):
        super().__init__('maxer', my_initial_card_ids, full_game_deck_ref_df)

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """Selects the field with the highest value on the agent's top card."""
        if not available_fields:
            # Handle edge case where no fields are available
            print("WARN MaxerAgent: No available fields!")
            return "Field0" # Must return something

        # Get values for available fields, coercing errors and filling NaNs
        field_values = pd.to_numeric(my_top_card_df[available_fields].iloc[0], errors='coerce').fillna(-np.inf)

        # Handle edge case where no valid numeric values exist
        if field_values.empty or (field_values == -np.inf).all():
             print("WARN MaxerAgent: No valid field values found, choosing random.")
             # Fallback to random choice if no max can be determined
             return random.choice(available_fields)

        # Find the index (which corresponds to the field name via available_fields) of the max value
        best_field_index_num = field_values.argmax()
        return available_fields[best_field_index_num]

    # update_state handled by base

class RanderAgent(Agent):
    """Agent that chooses a field completely randomly."""
    def __init__(self, my_initial_card_ids: Set[str], full_game_deck_ref_df: Optional[pd.DataFrame]):
        super().__init__('rander', my_initial_card_ids, full_game_deck_ref_df)

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """Selects a random field from the available options."""
        if not available_fields:
            print("WARN RanderAgent: No available fields!")
            return "Field0" # Need a default fallback
        return random.choice(available_fields)

    # update_state handled by base

class RandMaxerAgent(Agent):
    """With probability `randmax_fraction`, acts like Maxer, otherwise acts like Rander."""
    def __init__(self, my_initial_card_ids: Set[str], full_game_deck_ref_df: Optional[pd.DataFrame], randmax_fraction: float):
        super().__init__('randmaxer', my_initial_card_ids, full_game_deck_ref_df)
        if not (0 <= randmax_fraction <= 1):
            raise ValueError("randmax_fraction must be between 0 and 1")
        self.randmax_fraction: float = randmax_fraction

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """Randomly decides between using Maxer logic and Rander logic."""
        if not available_fields:
            return "Field0" # Handle empty case

        # Decide strategy based on probability
        if random.uniform(0, 1) < self.randmax_fraction:
            # --- Use Maxer logic ---
            field_values = pd.to_numeric(my_top_card_df[available_fields].iloc[0], errors='coerce').fillna(-np.inf)
            # Fallback if no valid values
            if field_values.empty or (field_values == -np.inf).all():
                 return random.choice(available_fields)
            # Return field with max value
            return available_fields[field_values.argmax()]
        else:
            # --- Use Rander logic ---
            return random.choice(available_fields)

    # update_state handled by base

# ==============================
# --- Agents Requiring Internal Calculation / State ---
# ==============================

class MeanerMaxAgent(Agent):
    """
    Agent that chooses the field with the maximum *normalized* value (z-score).
    Normalization is calculated internally during initialization using the reference deck.
    """
    def __init__(self,
                 my_initial_card_ids: Set[str],
                 full_game_deck_ref_df: pd.DataFrame # Requires the ref deck
                 ):
        super().__init__('meanermax', my_initial_card_ids, full_game_deck_ref_df)
        # Validation performed in base class for ref deck existence and columns

        # Calculate and store normalized data internally
        # Uses the NON-INDEXED copy stored in the base class for calculation
        self._normed_data_internal: Optional[pd.DataFrame] = self._calculate_normalization(self.full_game_deck_ref_df)

        # Create and store an indexed version of the normalized data for faster lookups
        self._indexed_normed_data: Optional[pd.DataFrame] = None
        if self._normed_data_internal is not None:
             if 'id' in self._normed_data_internal.columns:
                 try:
                      # Index a copy to preserve the original normalized data if needed
                      self._indexed_normed_data = self._normed_data_internal.copy()
                      self._indexed_normed_data.set_index('id', inplace=True, drop=False)
                 except Exception as error:
                      print(f"WARN MeanerMax: Error indexing normed data: {error}. Using non-indexed.")
                      self._indexed_normed_data = self._normed_data_internal # Fallback
             else:
                 # This case means 'id' column was somehow missing AFTER normalization
                 print("WARN MeanerMax: 'id' column not found in normalized data. Using non-indexed.")
                 self._indexed_normed_data = self._normed_data_internal

    def _calculate_normalization(self, deck_to_normalize: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates z-score normalization for numeric fields."""
        if deck_to_normalize is None: # Should be caught by __init__ check but be safe
             return None
        try:
            # Select only numeric fields for normalization
            numeric_columns = deck_to_normalize[self.field_colnames].select_dtypes(include=np.number)
            if numeric_columns.empty:
                print("Warn: MeanerMaxAgent found no numeric fields to normalize.")
                return None # Return None if no fields to normalize

            # Create a copy to store results
            normalized_data = deck_to_normalize.copy()
            scaler = StandardScaler()

            # Identify columns with non-zero (or near non-zero) standard deviation
            standard_deviations = numeric_columns.std()
            columns_to_scale = standard_deviations[standard_deviations > 1e-6].index.tolist()

            # Check if any columns are suitable for scaling
            if not columns_to_scale:
                print("Warn: MeanerMaxAgent found no fields with sufficient std dev to normalize.")
                # Return the original (copied) data if no scaling needed/possible
                return normalized_data

            # Log if some fields were skipped
            if len(columns_to_scale) < len(self.field_colnames):
                 skipped_columns = set(self.field_colnames) - set(columns_to_scale)
                 print(f"Warn: MeanerMaxAgent skipping normalization for fields (low/zero std dev): {skipped_columns}")

            # Apply StandardScaler transformation only to suitable columns
            normalized_data[columns_to_scale] = scaler.fit_transform(numeric_columns[columns_to_scale])
            #print("MeanerMaxAgent initialized internal normalized data.") # Less verbose
            return normalized_data

        except Exception as error:
            print(f"Warn: MeanerMaxAgent failed during internal normalization: {error}.")
            return None # Return None on failure

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """Chooses field based on max normalized value using internal data."""
        # Define a fallback strategy (Maxer)
        def maxer_fallback(fields_list):
            if not fields_list: return "Field0"
            vals = pd.to_numeric(my_top_card_df[fields_list].iloc[0], errors='coerce').fillna(-np.inf)
            if vals.empty or (vals == -np.inf).all(): return random.choice(fields_list)
            return fields_list[vals.argmax()]

        if not available_fields: return "Field0" # Handle edge case

        # Use the indexed normalized data for lookup if available, otherwise the non-indexed one
        normed_lookup_data = self._indexed_normed_data if self._indexed_normed_data is not None else self._normed_data_internal
        if normed_lookup_data is None:
            print("Warn: MeanerMaxAgent using Maxer strategy (internal normalization data unavailable).")
            return maxer_fallback(available_fields)

        card_id = my_top_card_df['id'].iloc[0]
        try:
            # Use .loc[] which works on both indexed and non-indexed (though faster if indexed)
            if isinstance(normed_lookup_data.index, pd.Index) and card_id in normed_lookup_data.index:
                 normalized_card_series = normed_lookup_data.loc[card_id]
            else: # Fallback to boolean mask if not indexed or ID missing (less efficient)
                 normalized_card_series = normed_lookup_data[normed_lookup_data['id'] == card_id].iloc[0]

            # Get normalized values for available fields, handle errors/NaNs
            normalized_values = pd.to_numeric(normalized_card_series[available_fields], errors='coerce').fillna(-np.inf)

            # Handle edge case: no valid normalized values found
            if normalized_values.empty or (normalized_values == -np.inf).all():
                 print(f"Warn: MeanerMaxAgent found no valid normalized values for card '{card_id}'. Choosing random.")
                 return random.choice(available_fields)

            # Return field corresponding to the max normalized value
            return available_fields[normalized_values.argmax()]

        except (KeyError, IndexError): # Card ID not found
             print(f"Warn: MeanerMaxAgent card ID '{card_id}' not found in normalized data. Using Maxer fallback.")
             return maxer_fallback(available_fields)
        except Exception as error: # Catch other potential lookup errors
             print(f"Warn: MeanerMaxAgent error during lookup for '{card_id}': {error}. Using Maxer fallback.")
             return maxer_fallback(available_fields)

    # update_state handled by base

class EXPertAgent(Agent):
    """
    Agent that estimates win/draw probability against the opponent's *tracked* hand set.
    Uses DataFrame slicing on an internally indexed reference deck for calculations.
    """
    def __init__(self,
                 my_initial_card_ids: Set[str],
                 full_game_deck_ref_df: pd.DataFrame,
                 EXPert_m_value: float = 0.5 # Renamed param for clarity
                ):
        """Initializes EXPertAgent, derives opponent initial state."""
        super().__init__('EXPert', my_initial_card_ids, full_game_deck_ref_df)

        # Validation
        if not (0 <= EXPert_m_value <= 1):
            raise ValueError("EXPert_m_value must be between 0 and 1")
        # Use the indexed ref df created and stored by the base class
        if self._indexed_ref_df is None:
            # Base class already tried to index, if it's still None, raise error
            raise ValueError("EXPertAgent requires an indexed reference deck (failed in base Agent?).")
        if not self.field_colnames:
            raise ValueError("EXPertAgent requires field column names (from base Agent).")

        self.m_factor: float = EXPert_m_value # Store the draw valuation factor

        # Base class init already ensures Int64 in self.full_game_deck_ref_df
        # and creates self._indexed_ref_df from that.

        # --- Derive Initial Opponent IDs ---
        # Get all possible IDs from the indexed reference deck's index
        all_game_ids: Set[str] = set(self._indexed_ref_df.index)
        # self.my_current_card_ids holds the agent's initial set from base __init__
        # Verify agent's initial IDs are valid (should be done by factory/main ideally)
        if not self.my_current_card_ids.issubset(all_game_ids):
             unmatched_ids = self.my_current_card_ids - all_game_ids
             print(f"Warning: EXPertAgent initial IDs not found in reference deck: {unmatched_ids}")
             # Correct internal state to only include valid IDs
             self.my_current_card_ids = self.my_current_card_ids.intersection(all_game_ids)

        # Calculate opponent's initial set by subtracting own set from total set
        derived_opponent_initial_ids: Set[str] = all_game_ids - self.my_current_card_ids
        #print(f"EXPert Debug: Derived {len(derived_opponent_initial_ids)} initial opponent IDs.") # Less verbose

        # --- Internal Opponent Hand Tracking (Stores only the set of IDs) ---
        self.tracked_opponent_card_ids: Set[str] = derived_opponent_initial_ids.copy()


    # Override update_state to include opponent tracking
    def update_state(self,
                     outcome: str,
                     chosen_field: str,
                     my_card_played_df: pd.DataFrame,
                     opponent_actual_card_id: Optional[str],
                     opponent_revealed_value: Any, # Placeholder, not used by this simple tracking
                     pot_cards_df: pd.DataFrame,
                     cards_i_won_df: pd.DataFrame,
                     cards_opponent_won_df: pd.DataFrame):
        """Updates own card set (via super) and opponent's tracked card set."""
        # --- 1. Update own deck (essential!) ---
        # This correctly removes my_card_played_df ID and adds cards_i_won_df IDs
        super().update_state(outcome, chosen_field, my_card_played_df, opponent_actual_card_id, opponent_revealed_value, pot_cards_df, cards_i_won_df, cards_opponent_won_df)

        # --- 2. Update opponent tracking state ---
        my_played_id = my_card_played_df['id'].iloc[0] if not my_card_played_df.empty else None
        opponent_played_id = opponent_actual_card_id # Revealed on win/draw

        # A. Remove opponent's played card from tracking *if* we know its ID (win/draw)
        if opponent_played_id:
            self.tracked_opponent_card_ids.discard(opponent_played_id)

        # B. If we lost, opponent gained cards (our played card + pot)
        if outcome == 'loss':
            # Add our played card ID to their tracked set
            if my_played_id:
                self.tracked_opponent_card_ids.add(my_played_id)
            # Add all IDs from cards_opponent_won_df (this includes our played card + pot)
            # Using update handles potential duplicates correctly.
            if not cards_opponent_won_df.empty and 'id' in cards_opponent_won_df.columns:
                 opponent_gained_ids = set(cards_opponent_won_df['id'])
                 self.tracked_opponent_card_ids.update(opponent_gained_ids)

        # C. If we won, opponent lost the cards *we* won
        #    (which are opponent's played card + pot cards)
        elif outcome == 'win':
            # cards_i_won_df contains opponent's played card + pot cards.
            # Remove all IDs present in cards_i_won_df from opponent's tracked set.
            # This handles removing their played card (even if ID wasn't known before but was in pot)
            # and removing their card from the pot.
            if not cards_i_won_df.empty and 'id' in cards_i_won_df.columns:
                we_won_ids = set(cards_i_won_df['id'])
                self.tracked_opponent_card_ids.difference_update(we_won_ids)

        # D. If draw, opponent's played card left their hand for the pot.
        #    This is handled by step A above (discarding opponent_played_id).

    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """
        Chooses field maximizing expected score against the tracked opponent hand set.
        Uses DataFrame slicing for calculation after initial indexed lookup.
        """
        # Define fallback strategy (Maxer)
        def maxer_fallback(fields_list):
             if not fields_list: return "Field0"
             vals = pd.to_numeric(my_top_card_df[fields_list].iloc[0], errors='coerce').fillna(-np.inf)
             if vals.empty or (vals == -np.inf).all(): return random.choice(fields_list)
             return fields_list[vals.argmax()]

        if not available_fields: return "Field0" # Handle no fields

        # Use the internally tracked opponent hand ID set
        current_tracked_opponent_ids = self.tracked_opponent_card_ids
        if not current_tracked_opponent_ids:
            # If tracking is empty (e.g., opponent out of cards, or error), fallback
            return maxer_fallback(available_fields)

        # --- Get opponent stats using DataFrame slicing (via indexed lookup) ---
        opponent_stats_df = pd.DataFrame() # Initialize empty
        try:
            # Convert set to list for .loc indexing
            opponent_ids_list = list(current_tracked_opponent_ids)
            # Use the indexed ref df stored in the base class
            # This slice contains stats for all currently tracked opponent cards
            opponent_stats_df = self._indexed_ref_df.loc[opponent_ids_list]
        except KeyError:
             # Attempt to recover if some tracked IDs became invalid
             valid_opponent_ids = list(current_tracked_opponent_ids.intersection(self._indexed_ref_df.index))
             if valid_opponent_ids:
                 print(f"WARN EXPert: Some tracked opponent IDs not found in ref index. Using subset.")
                 try:
                      opponent_stats_df = self._indexed_ref_df.loc[valid_opponent_ids]
                 except Exception as inner_error: # Handle error during subset lookup too
                      print(f"WARN EXPert: Error looking up opponent stats subset: {inner_error}. Maxer fallback.")
                      return maxer_fallback(available_fields)
             else:
                print(f"WARN EXPert: No valid tracked opponent IDs found in ref index. Maxer fallback.")
                return maxer_fallback(available_fields)
        except Exception as error:
             # Catch other unexpected errors during lookup
             print(f"WARN EXPert: Error looking up opponent stats: {error}. Maxer fallback.")
             return maxer_fallback(available_fields)

        # If slicing resulted in an empty DataFrame (e.g., all IDs were invalid)
        if opponent_stats_df.empty:
             print(f"WARN EXPert: Opponent stats slice empty. Maxer fallback.")
             return maxer_fallback(available_fields)

        # Get my card's values once, ensure Int64
        my_field_values = pd.to_numeric(my_top_card_df[available_fields].iloc[0], errors='coerce').astype('Int64')
        expected_scores = []
        number_of_tracked_opp_cards = len(opponent_stats_df) # Use length of the actual stats DF

        # Iterate through each available field to calculate expected score
        for field_index, field_name in enumerate(available_fields):
            my_value = my_field_values.iloc[field_index]
            score = -1.0 # Default score if calculation isn't possible

            # Calculate score only if my value is valid
            if not pd.isna(my_value) and number_of_tracked_opp_cards > 0:
                try:
                    # Get opponent values for this field from the sliced DataFrame
                    opponent_values = pd.to_numeric(opponent_stats_df[field_name], errors='coerce').astype('Int64')

                    # Calculate wins and draws using vectorized operations
                    wins = (opponent_values < my_value).sum()
                    draws = (opponent_values == my_value).sum()
                    score = (wins + self.m_factor * draws) / number_of_tracked_opp_cards
                except KeyError: # Field might be missing from opponent_stats_df? Should not happen.
                    print(f"WARN EXPert: Field '{field_name}' not found in opponent stats slice.")
                    score = -1.0 # Assign invalid score
                except Exception as error:
                     print(f"WARN EXPert: Error calculating score for field '{field_name}': {error}")
                     score = -1.0 # Assign invalid score

            expected_scores.append(score)

        # --- Choose best field based on calculated scores ---
        if not expected_scores: # If loop didn't run
             return random.choice(available_fields) if available_fields else "Field0"

        max_score = np.max(expected_scores)

        # Find all fields achieving the maximum score (within tolerance), ensuring score is valid
        best_field_indices = [
            index for index, current_score in enumerate(expected_scores)
            if current_score >= -0.5 and np.isclose(current_score, max_score) # Check valid score and max
        ]

        # If no valid best field found (e.g., all scores were -1), use fallback
        if not best_field_indices:
            return maxer_fallback(available_fields)

        # Randomly choose among the best fields to break ties
        chosen_field_name = available_fields[random.choice(best_field_indices)]
        return chosen_field_name


class GoliathAgent(Agent):
    """
    Advanced agent tracking possible opponent card identities and their order.
    Uses DataFrame slicing on indexed ref deck for internal calculations.
    """
    def __init__(self,
                 my_initial_card_ids: Set[str],
                 full_game_deck_ref_df: pd.DataFrame,
                 EXPert_m_value: float = 0.5, # Renamed param
                 debug_mode: bool = False):
        """Initializes Goliath, derives opponent initial state."""
        super().__init__('Goliath', my_initial_card_ids, full_game_deck_ref_df)
        # Validation
        if not (0 <= EXPert_m_value <= 1): raise ValueError("Goliath EXPert_m_value 0-1")
        if self._indexed_ref_df is None: raise ValueError("Goliath requires indexed ref deck.")
        if not self.field_colnames: raise ValueError("Goliath requires fields.")
        self.m_factor = EXPert_m_value # Store draw factor

        # Base class init already ensures Int64 in self.full_game_deck_ref_df
        # and creates self._indexed_ref_df. No need to repeat Int64 conversion here.

        # --- Derive Initial Opponent IDs ---
        all_ids = set(self._indexed_ref_df.index)
        my_ids = self.my_current_card_ids # From base init
        if not my_ids.issubset(all_ids): self.my_current_card_ids = my_ids.intersection(all_ids)
        derived_opponent_initial_ids: Set[str] = all_ids - self.my_current_card_ids

        # --- State Tracking Initialization ---
        # List where each element represents a card slot in the opponent's hand (top to bottom).
        # Element is either: str (known ID) or Set[str] (possible IDs).
        self.opponent_hand_state: List[Union[str, Set[str]]] = []
        if derived_opponent_initial_ids:
            # Start by assuming any derived opponent card could be in any position
            self.opponent_hand_state = [derived_opponent_initial_ids.copy() for _ in range(len(derived_opponent_initial_ids))]
        else:
            # This case should only happen if the game starts with one player having all cards
             print("Warning: GoliathAgent init: No opponent IDs derived! State will be empty.")

        # --- Debugging ---
        self._debug = debug_mode
        self._log_init(len(self.opponent_hand_state), len(derived_opponent_initial_ids))

    # --- Internal Helper Methods ---
    def _log(self, *args):
        """Prints debug messages if debug_mode is True."""
        if self._debug:
            # Show current length of opponent state list for context
            print(f"GOL ({len(self.opponent_hand_state)} opp slots):", *args)

    def _log_init(self, opponent_card_count: int, initial_possibility_count: int):
         """Logs initial state information."""
         if self._debug:
              print(f"GOL INIT: Opponent starts with {opponent_card_count} cards.")
              if opponent_card_count > 0:
                   # The initial count represents the number of cards opponent could have
                   print(f"GOL INIT: Initial possibilities/card = {initial_possibility_count}")

    def _print_state(self, context_message: str = "State"):
        """Prints the current opponent hand state for debugging."""
        if self._debug:
            print(f"GOL STATE --- {context_message} (len={len(self.opponent_hand_state)}) ---")
            # Use pprint for better readability of the list containing sets/strings
            pprint.pprint(self.opponent_hand_state, depth=2, width=150) # Limit depth for large sets
            print(f"GOL STATE --- End {context_message} ---")

    def _propagate(self, known_card_id: str, excluded_index: int = -1):
        """
        Propagates the knowledge of a known card ID. Removes this ID as a
        possibility from all other slots containing sets. If any set collapses
        to a single possibility, recursively propagates that new knowledge.

        Args:
            known_card_id: The ID of the card whose position/removal is now known.
            excluded_index: The index in opponent_hand_state where this ID is confirmed
                           (or -1 if the card left opponent's hand entirely).
                           This index's state itself is not modified by removing this ID.
        """
        # Use a queue for breadth-first propagation to handle cascading deductions
        propagation_queue: Deque[tuple[str, int]] = deque([(known_card_id, excluded_index)])
        # Keep track of IDs processed in *this specific propagation chain* to prevent cycles
        processed_in_this_call: Set[str] = set()

        # self._log(f"Propagate Start: ID='{known_card_id}', ExcludedIdx={excluded_index}") # Optional verbose log

        while propagation_queue:
            target_id_to_remove, source_info_index = propagation_queue.popleft()

            # Avoid reprocessing if already handled in this chain
            if target_id_to_remove in processed_in_this_call:
                continue
            processed_in_this_call.add(target_id_to_remove)
            # self._log(f" Propagating remove '{target_id_to_remove}' (info from idx {source_info_index})...") # Optional verbose log

            newly_identified_cards: List[tuple[str, int]] = [] # Store (id, index) of newly known cards

            # Iterate through the opponent hand state list by index
            for current_index in range(len(self.opponent_hand_state)):
                 # Don't modify the state at the excluded index based on this specific propagation trigger
                 if current_index == source_info_index:
                     continue

                 current_slot_state = self.opponent_hand_state[current_index]

                 # Only attempt removal if the slot holds a set of possibilities
                 if isinstance(current_slot_state, set):
                     # Check if the ID to remove is present in this set
                     if target_id_to_remove in current_slot_state:
                         current_slot_state.discard(target_id_to_remove) # Remove the possibility

                         # Check if this removal collapsed the set to a single known ID
                         if len(current_slot_state) == 1:
                             newly_known_card_id = list(current_slot_state)[0]
                             # Update the state list to store the known string ID
                             self.opponent_hand_state[current_index] = newly_known_card_id
                             self._log(f"  Propagate COLLAPSED index {current_index} -> '{newly_known_card_id}'!")
                             # If this newly identified card hasn't been processed yet in this chain,
                             # add it to the queue for further propagation.
                             if newly_known_card_id not in processed_in_this_call:
                                 newly_identified_cards.append((newly_known_card_id, current_index))
                         elif len(current_slot_state) == 0:
                             # This indicates a contradiction in state tracking
                             print(f"CRITICAL WARNING: Goliath state set at index {current_index} became empty removing {target_id_to_remove}.")
                             # The state remains an empty set, potentially causing issues later.

            # Add any newly collapsed IDs to the queue for the next level of propagation
            if newly_identified_cards:
                # self._log(f"  Adding newly identified cards to propagation queue: {newly_identified_cards}") # Optional verbose log
                propagation_queue.extend(newly_identified_cards)

        # self._log(f"Propagate End: Finished chain for '{known_card_id}'") # Optional verbose log

    def _filter(self, possible_ids: Set[str], field_name: str, revealed_value: Any) -> Set[str]:
        """
        Filters a set of possible card IDs based on a revealed value for a specific field.
        Uses DataFrame slicing on indexed reference deck for lookup.

        Args:
            possible_ids: The set of card IDs to filter.
            field_name: The field name (string) to check.
            revealed_value: The value observed for the opponent's card on that field.

        Returns:
            A new set containing only the IDs from possible_ids that match the
            revealed value in the given field. Returns empty set on errors.
        """
        if not possible_ids:
            return set()

        # Convert revealed value to integer for comparison
        try:
            value_to_match = int(np.round(pd.to_numeric(revealed_value)))
        except (ValueError, TypeError, OverflowError) as error:
             # Log error and return empty set if value is unusable
             print(f"WARN Goliath Filter: Cannot compare revealed value '{revealed_value}': {error}")
             return set() # Cannot filter with invalid value

        try:
            # Use indexed lookup on the reference DataFrame for efficiency
            ids_list = list(possible_ids)
            # Retrieve only the relevant field column for the possible IDs
            subset_df = self._indexed_ref_df.loc[ids_list, [field_name]] # Use .loc for index lookup
        except KeyError:
             # Handle case where some possible_ids might not be in index
             valid_ids = list(possible_ids.intersection(self._indexed_ref_df.index))
             if not valid_ids: return set()
             print(f"WARN Goliath Filter: Some possible IDs not in ref index. Filtering subset.")
             try:
                 subset_df = self._indexed_ref_df.loc[valid_ids, [field_name]]
             except Exception as inner_error:
                  print(f"WARN Goliath Filter: Error looking up subset stats: {inner_error}.")
                  return set()
        except Exception as error:
             print(f"WARN Goliath Filter: Error looking up stats for IDs {possible_ids}: {error}")
             return set() # Return empty on lookup error

        if subset_df.empty:
            return set()

        # Perform comparison using pandas Series operations
        # Ensure reference values are also Int64 for reliable comparison
        reference_values = pd.to_numeric(subset_df[field_name], errors='coerce').astype('Int64')
        matching_mask = (reference_values == value_to_match)

        # Return the set of IDs where the value matched
        # Need to get original IDs corresponding to the True values in the mask
        matching_card_ids = set(subset_df.loc[matching_mask.fillna(False)].index)
        return matching_card_ids


    def choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str:
        """
        Chooses field using expected score based on possibilities for opponent's top card.
        Uses DataFrame slicing on indexed ref deck for calculations.
        """
        # Define fallback strategy
        def maxer_fallback(fields_list):
            if not fields_list: return "Field0"
            vals = pd.to_numeric(my_top_card_df[fields_list].iloc[0], errors='coerce').fillna(-np.inf)
            if vals.empty or (vals == -np.inf).all(): return random.choice(fields_list)
            return fields_list[vals.argmax()]

        # Basic validation
        if not self.opponent_hand_state:
            print("Warn: Goliath choosing, opponent state empty. Using Maxer fallback.")
            return maxer_fallback(available_fields)
        if not available_fields:
            raise ValueError("Goliath choose_field called with no available fields.")

        # Determine the set of possible IDs for the opponent's top card
        top_card_state_info = self.opponent_hand_state[0]
        possible_opponent_top_ids: Set[str] = set()
        if isinstance(top_card_state_info, str): # ID is known
            possible_opponent_top_ids = {top_card_state_info}
        elif isinstance(top_card_state_info, set): # Set of possibilities
            possible_opponent_top_ids = top_card_state_info
        else:
            raise TypeError(f"Invalid state type at opponent_hand_state[0]: {type(top_card_state_info)}")

        # Check if the possibility set became empty (state contradiction)
        if not possible_opponent_top_ids:
            print("CRITICAL Warn: GoliathAgent state[0] has no possibilities. Using Maxer fallback.")
            return maxer_fallback(available_fields)

        # --- Get opponent stats using DataFrame slicing (via indexed lookup) ---
        opponent_stats_df = pd.DataFrame() # Initialize empty
        try:
            ids_list = list(possible_opponent_top_ids)
            # Use the indexed ref df stored in the base class
            opponent_stats_df = self._indexed_ref_df.loc[ids_list]
        except KeyError:
             # Handle if some possible IDs are not in index
             valid_ids = list(possible_opponent_top_ids.intersection(self._indexed_ref_df.index))
             if not valid_ids: print("CRIT ERR: No valid possible opp IDs found in ref index. Maxer."); return maxer_fallback(available_fields)
             print(f"WARN Goliath Choose: Some possible IDs not found. Using subset.")
             try: opponent_stats_df = self._indexed_ref_df.loc[valid_ids]
             except Exception as inner_error: print(f"WARN Goliath Choose: Error looking up subset: {inner_error}. Maxer fallback."); return maxer_fallback(available_fields)
        except Exception as error:
             print(f"WARN Goliath Choose: Error looking up opponent stats: {error}. Maxer fallback.")
             return maxer_fallback(available_fields)

        # Check if slicing resulted in an empty DataFrame
        if opponent_stats_df.empty:
             print("CRIT Warn: Opponent possibility slice empty. Maxer fallback.")
             return maxer_fallback(available_fields)

        # --- Calculate Expected Scores ---
        my_field_values = pd.to_numeric(my_top_card_df[available_fields].iloc[0], errors='coerce').astype('Int64')
        expected_scores = []
        number_of_possibilities = len(opponent_stats_df) # Use length of actual stats DF

        for field_index, field_name in enumerate(available_fields):
            my_value = my_field_values.iloc[field_index]
            score = -1.0 # Default score if calculation fails

            if not pd.isna(my_value) and number_of_possibilities > 0:
                try:
                    # Get opponent values for this field, ensure Int64
                    opponent_values = pd.to_numeric(opponent_stats_df[field_name], errors='coerce').astype('Int64')
                    # Calculate wins and draws using vectorized operations
                    win_count = (opponent_values < my_value).sum()
                    draw_count = (opponent_values == my_value).sum()
                    score = (win_count + self.m_factor * draw_count) / number_of_possibilities
                except KeyError: score = -1.0 # Should not happen if field_name is valid
                except Exception as error: print(f"WARN Goliath Choose: Error calc score field '{field_name}': {error}"); score = -1.0

            expected_scores.append(score)

        # --- Choose Best Field ---
        if not expected_scores: return random.choice(available_fields) if available_fields else "Field0"

        max_score = np.max(expected_scores)

        # Find indices of fields achieving the max score (must be valid score >= 0)
        best_field_indices = [
            idx for idx, current_score in enumerate(expected_scores)
            if current_score >= -0.0001 and np.isclose(current_score, max_score) # Allow for float precision near 0
        ]

        # If no valid best field found (e.g., all scores were -1), use fallback
        if not best_field_indices:
            return maxer_fallback(available_fields)

        # Randomly choose among the best fields
        chosen_field_name = available_fields[random.choice(best_field_indices)]
        return chosen_field_name


    def update_state(self,
                     outcome: str,
                     chosen_field: str,
                     my_card_played_df: pd.DataFrame,
                     opponent_actual_card_id: Optional[str],
                     opponent_revealed_value: Any,
                     pot_cards_df: pd.DataFrame,
                     cards_i_won_df: pd.DataFrame,
                     cards_opponent_won_df: pd.DataFrame):
        """Updates own card set (via super) and Goliath's opponent state list."""
        # --- 1. Update own deck (essential!) ---
        super().update_state(outcome, chosen_field, my_card_played_df, opponent_actual_card_id, opponent_revealed_value, pot_cards_df, cards_i_won_df, cards_opponent_won_df)

        # --- 2. Update opponent tracking state (Goliath's specific logic) ---
        opponent_id = opponent_actual_card_id # Use clearer name
        opponent_value = opponent_revealed_value
        opponent_won_df = cards_opponent_won_df # Use clearer name

        # self._log(f"\nGoliath Update Start: Outcome={outcome}, OppVal={opponent_value}, OppID='{opponent_id or 'UNK'}'") # Less Verbose
        # self._print_state("Update Entry State") # Less Verbose

        current_opponent_state_length = len(self.opponent_hand_state)
        if current_opponent_state_length == 0:
            # Opponent was empty, update only if they won cards
            if outcome == 'loss' and not opponent_won_df.empty:
                won_ids_list = opponent_won_df['id'].tolist()
                self._log(f" Opponent was empty, now gains known cards: {won_ids_list}")
                self.opponent_hand_state = won_ids_list # State is now list of known IDs
            # self._print_state("Update End (Opponent Started/Remained Empty)") # Less verbose
            return

        # --- Get state of the card opponent played (BEFORE modifying the list) ---
        top_card_state_before_update = self.opponent_hand_state[0]

        # --- Apply Value Filter ---
        known_id_from_filter: Optional[str] = None
        state_after_value_filtering: Union[str, Set[str]] = top_card_state_before_update # Default

        if isinstance(top_card_state_before_update, set):
            try:
                matching_ids: Set[str] = self._filter(top_card_state_before_update, chosen_field, opponent_value)
                # Check if filter reduced possibilities
                if len(matching_ids) < len(top_card_state_before_update):
                    state_after_value_filtering = matching_ids # Update state var
                    self.opponent_hand_state[0] = matching_ids # Update live state list

                    if len(matching_ids) == 1: # Collapsed to one possibility
                        known_id_from_filter = list(matching_ids)[0]
                        self.opponent_hand_state[0] = known_id_from_filter # Collapse in list
                        state_after_value_filtering = known_id_from_filter # Update state var
                        self._log(f"  >>> Value Filter COLLAPSED state[0] to '{known_id_from_filter}'!")
                        self._propagate(known_id_from_filter, excluded_index=0) # Propagate new knowledge
                    elif len(matching_ids) == 0: # Contradiction
                         print(f"CRIT Warn: Goliath Filter contradiction. State[0] unchanged.")
                         # Revert state to avoid using empty set
                         state_after_value_filtering = top_card_state_before_update
                         self.opponent_hand_state[0] = top_card_state_before_update

            except ValueError as error: # Error during filtering (e.g., bad value)
                 print(f"Error during Goliath Value Filter: {error}. State[0] unchanged.")
                 state_after_value_filtering = top_card_state_before_update # Keep original on error


        # --- Apply Outcome Logic (Card movement and ID reveals) ---
        id_to_propagate_from_outcome: Optional[str] = None
        propagation_source_index: int = -1 # Default: card left play

        try:
            if outcome == 'win':
                # Opponent lost their top card to us. Remove it from state list.
                # We also learned their ID (opponent_id).
                if opponent_id is None: print("Warn: Goliath 'win' but opponent_id is None.")
                else: id_to_propagate_from_outcome = opponent_id; propagation_source_index = -1
                self.opponent_hand_state.pop(0) # Remove top element

            elif outcome == 'loss':
                # Opponent keeps their played card (state_after_value_filtering), rotates to back.
                # Opponent gains our played card + pot cards (opponent_won_df).
                state_to_rotate = state_after_value_filtering
                self.opponent_hand_state.pop(0) # Remove from top
                self.opponent_hand_state.append(state_to_rotate) # Add rotated state to bottom

                # Add newly won cards (known IDs) after the rotated card
                won_ids_list = opponent_won_df['id'].tolist() if not opponent_won_df.empty else []
                for card_id_won in won_ids_list:
                    self.opponent_hand_state.append(card_id_won) # Append known ID string
                    # Propagate this new known card's position
                    self._propagate(card_id_won, excluded_index=len(self.opponent_hand_state)-1)

            elif outcome == 'draw':
                # Opponent's played card leaves their hand for the pot. Remove from state list.
                # We learned their ID (opponent_id).
                if opponent_id is None: print("Warn: Goliath 'draw' but opponent_id is None.")
                else: id_to_propagate_from_outcome = opponent_id; propagation_source_index = -1
                self.opponent_hand_state.pop(0) # Remove top element

        except IndexError:
            print(f"CRIT Err: Goliath IndexError during outcome '{outcome}'. State list might be corrupted.")
        except Exception as error:
             print(f"CRIT Err during Goliath outcome logic '{outcome}': {type(error).__name__} - {error}")


        # --- Apply Final Propagation (if needed) ---
        if id_to_propagate_from_outcome is not None and id_to_propagate_from_outcome != known_id_from_filter:
            # self._log(f" Outcome requires propagation of known ID: '{id_to_propagate_from_outcome}'") # Less verbose
            self._propagate(id_to_propagate_from_outcome, excluded_index=propagation_source_index)

        # self._print_state(f"Update End {outcome}") # Less verbose

# ==============================
# --- Agent Factory Function ---
# ==============================
def create_agent(strategy: str,
                 # --- Data needed for agent INIT ---
                 my_initial_card_ids: Optional[Set[str]]=None,
                 full_game_deck_ref_df: Optional[pd.DataFrame]=None,
                 # --- Agent-specific params (use full names) ---
                 randmax_fraction: Optional[float]=None, # Changed name
                 EXPert_m_value: float=0.5,             # Changed name
                 goliath_debug_flag: bool=False,         # Changed name
                 # --- Context (usually ignored by agents) ---
                 number_of_fields: Optional[int]=None,   # Changed name
                 number_of_cards: Optional[int]=None     # Changed name
                 ) -> Agent:
    """
    Factory function to create agent instances based on strategy name.
    Passes minimal required initialization data (initial ID set, ref deck).
    Uses descriptive parameter names.
    """
    strategy_lower = strategy.lower()
    if my_initial_card_ids is None:
        raise ValueError(f"Agent strategy '{strategy}' requires 'my_initial_card_ids'.")

    # Prepare base arguments needed by all Agent constructors
    base_args = {
        'my_initial_card_ids': my_initial_card_ids,
        'full_game_deck_ref_df': full_game_deck_ref_df
    }

    # Select agent based on strategy name
    if strategy_lower == 'user':
        return UserAgent(**base_args)
    elif strategy_lower == 'maxer':
        return MaxerAgent(**base_args)
    elif strategy_lower == 'rander':
        return RanderAgent(**base_args)
    elif strategy_lower == 'randmaxer':
        if randmax_fraction is None:
            raise ValueError("RandMaxerAgent requires 'randmax_fraction'.")
        # Pass specific parameter in addition to base args
        return RandMaxerAgent(**base_args, randmax_fraction=randmax_fraction)
    elif strategy_lower == 'meanermax':
        if full_game_deck_ref_df is None:
            raise ValueError("MeanerMaxAgent requires 'full_game_deck_ref_df'.")
        # MeanerMax now only needs base_args (which includes the ref deck)
        return MeanerMaxAgent(**base_args)
    elif strategy_lower == 'expert':
        if full_game_deck_ref_df is None:
            raise ValueError("EXPertAgent requires 'full_game_deck_ref_df'.")
        # Pass specific parameter
        return EXPertAgent(**base_args, EXPert_m_value=EXPert_m_value)
    elif strategy_lower == 'goliath':
        if full_game_deck_ref_df is None:
            raise ValueError("GoliathAgent requires 'full_game_deck_ref_df'.")
        # Pass specific parameters
        return GoliathAgent(**base_args, EXPert_m_value=EXPert_m_value, debug_mode=goliath_debug_flag)

    # --- Add new agent creation logic here ---
    # elif strategy_lower == 'new_strategy':
    #     # Check for required parameters...
    #     # return NewStrategyAgent(**base_args, specific_param=some_value)

    else:
        # Handle unknown strategy
        raise ValueError(f"Unknown agent strategy provided: '{strategy}'")