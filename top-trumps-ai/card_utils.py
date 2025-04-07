# card_utils.py
"""
Utilities for generating Top Trumps card decks and related data structures.
Provides functions for creating unique card IDs and generating randomized decks.
"""
import random
import string
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional # For type hinting

def id_generator(size: int = 6, chars: str = string.ascii_uppercase + string.digits) -> str:
    """
    Generates a random alphanumeric ID string of a specified size.

    Args:
        size: The desired length of the ID string.
        chars: The character set to draw from for the ID.

    Returns:
        A randomly generated ID string.
    """
    return ''.join(random.choice(chars) for _ in range(size))

def generate_decks(number_of_cards: int = 50, number_of_fields: int = 5) -> Tuple[
    pd.DataFrame, pd.DataFrame, List[str], pd.DataFrame
]:
    """
    Generates a complete set of Top Trumps cards with random values,
    splits them into two decks, and returns the decks along with
    the full reference deck.

    Args:
        number_of_cards: Total number of cards to generate. Must be positive and even.
        number_of_fields: Number of numerical fields (attributes) per card. Must be positive.

    Returns:
        A tuple containing:
            - pd.DataFrame: Player 1's initial deck DataFrame.
            - pd.DataFrame: Player 2's initial deck DataFrame.
            - List[str]: List of the generated field column names (e.g., ['Field0', 'Field1']).
            - pd.DataFrame: A reference copy of the full deck with all cards and stats
                              (using Int64 for fields). Used internally by agents.

    Raises:
        ValueError: If number_of_cards or number_of_fields are invalid.
        RuntimeError: If unique ID generation fails within safety limits.
        TypeError: If numeric conversion of fields fails unexpectedly.
    """
    # --- Input Validation ---
    if not isinstance(number_of_cards, int) or number_of_cards <= 0 or number_of_cards % 2 != 0:
        raise ValueError("number_of_cards must be a positive, even integer.")
    if not isinstance(number_of_fields, int) or number_of_fields <= 0:
        raise ValueError("number_of_fields must be a positive integer.")

    # --- Generate Field Names ---
    # Create names like 'Field0', 'Field1', ...
    field_column_names: List[str] = [f'Field{index}' for index in range(number_of_fields)]

    # --- Generate Unique Card IDs ---
    card_ids_set: set[str] = set()
    generation_attempts: int = 0
    max_generation_attempts: int = number_of_cards * 10 # Safety limit

    # Loop until the desired number of unique IDs is generated or max attempts reached
    while len(card_ids_set) < number_of_cards and generation_attempts < max_generation_attempts:
        card_ids_set.add(id_generator())
        generation_attempts += 1

    # Check if ID generation was successful
    if len(card_ids_set) < number_of_cards:
        raise RuntimeError(f"Failed to generate {number_of_cards} unique IDs after {max_generation_attempts} attempts.")

    # Convert the set to a list and shuffle for randomness in initial deck dealing
    card_ids_list: List[str] = list(card_ids_set)
    random.shuffle(card_ids_list)

    # --- Generate Field Statistics (for card value distribution) ---
    # Define random means and standard deviations for each field
    field_means: np.ndarray = np.random.randint(low=30, high=90, size=number_of_fields)
    field_standard_deviations: np.ndarray = np.random.randint(low=5, high=20, size=number_of_fields)

    # --- Generate Raw Card Values ---
    # Create a dictionary to hold card data, starting with IDs
    card_data_dictionary: Dict[str, Any] = {'id': card_ids_list}
    # Generate values for each field using a normal distribution
    for i, column_name in enumerate(field_column_names):
        # Generate float values based on field's mean and standard deviation
        float_values: np.ndarray = np.random.normal(field_means[i], field_standard_deviations[i], number_of_cards)
        # Round to nearest integer, ensure non-negative
        card_data_dictionary[column_name] = [max(0.0, np.round(value)) for value in float_values]

    # --- Create Initial DataFrame from generated data ---
    full_cards_dataframe: pd.DataFrame = pd.DataFrame(card_data_dictionary)

    # --- Finalize Numeric Types (Convert fields to pandas' nullable Int64) ---
    # This ensures consistency and handles potential missing values gracefully.
    safe_fill_value_for_nan: int = -1 # Use a value unlikely to be a valid stat
    for column_name in field_column_names:
        try:
            # Convert column to numeric first, coercing errors to NaN
            numeric_column = pd.to_numeric(full_cards_dataframe[column_name], errors='coerce')
            # Fill any NaNs that resulted from coercion before rounding and casting
            filled_column = numeric_column.fillna(safe_fill_value_for_nan)
            # Round to nearest integer and cast to nullable Int64 type
            full_cards_dataframe[column_name] = filled_column.round().astype('Int64')

            # Optionally warn if NaNs were actually present and filled
            if numeric_column.isna().any():
                 print(f"Warning: NaNs found during Int64 conversion for column '{column_name}'. Filled with {safe_fill_value_for_nan}.")

        except Exception as error:
            # Catch potential errors during conversion process
            raise TypeError(f"Failed to convert column '{column_name}' to Int64: {error}")

    # --- Create Reference Deck (Immutable copy for agent use) ---
    # Agents needing full game info will use this reference.
    full_game_deck_reference_df: pd.DataFrame = full_cards_dataframe.copy()

    # --- Normalization logic removed - now handled internally by agents needing it ---

    # --- Shuffle and Split the Finalized Full Deck for Gameplay ---
    # Shuffle the rows randomly
    shuffled_cards_df: pd.DataFrame = full_cards_dataframe.sample(frac=1, random_state=np.random.RandomState()).reset_index(drop=True)
    # Find the midpoint to split the deck
    split_point: int = number_of_cards // 2
    # Create the initial decks for Player 1 and Player 2
    player1_initial_deck_df: pd.DataFrame = shuffled_cards_df.iloc[:split_point].copy()
    player2_initial_deck_df: pd.DataFrame = shuffled_cards_df.iloc[split_point:].copy()

    # --- Return all necessary components ---
    return (
        player1_initial_deck_df,
        player2_initial_deck_df,
        field_column_names,
        full_game_deck_reference_df # Reference deck is the last item now
    )