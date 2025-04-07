# Top Trumps AI Simulation

## Overview

This project provides a simulation environment for developing and testing AI agents that play the card game Top Trumps. It allows users to implement different strategies, pit them against each other in simulations, and analyze their performance.

Separation between the game logic and the agent's decision-making process is enforced by the `GameEngine`, whereby agents must treat the engine as an API that provides limited information according to defined rules.

## Game Rules

This simulation implements the following Top Trumps rules:

1.  **Setup:**
    * Two players (Player 1 and Player 2).
    * A deck of cards with unique IDs and several numerical fields (attributes) is generated. Players may look through this deck.
    * The deck is shuffled and dealt evenly between the two players. If there's an odd number initially, one player might start with one more card.
    * Each player is allowed to look through their own initial half of the deck when dealt to them, observing what cards they received from the split. The player decks are then individually shuffled before gameplay begins.

2.  **Gameplay:**
    * The game proceeds in tricks (turns).
    * Player 1 starts the first trick.
    * For subsequent tricks, the **winner** of the previous trick starts the next one.
    * If the previous trick was a **draw**, the **same player** who started the drawn trick starts the next trick.

3.  **Playing a Trick:**
    * The player starting the trick (the "starter") looks at their **top card** (that is, the card currently at the top of their deck).
    * The starter **chooses one field** (attribute) from that card.
    * The numerical values on the **chosen field** are compared, between the starter's top card and the other player's top card.

4.  **Winning a Trick:**
    * The player whose card has the **higher value** on the chosen field wins the trick.
    * The winner takes **both** played cards (their own and the opponent's) **plus** any cards currently accumulated in the central "draw pile" (see Draws).
    * These collected cards are then added to the **bottom** of the winner's deck in a specific order:
        1.  The winner's own card that was just played.
        2.  The loser's card that was just played.
        3.  The cards from the draw pile (if any), in the order they were originally added to the pile.

5.  **Losing a Trick:**
    * The player whose card has the **lower value** loses the trick.
    * Their played card is taken by the winner (as described above).
    * They do not learn the exact identity of the winner's top card (beyond the field of comparison, and the numerical value of that card on that field, which were announced).

6.  **Drawing a Trick:**
    * If both players' top cards have the **same value** on the chosen field, the trick is a draw.
    * **Both** played cards are placed into a central "draw pile".
    * The order cards are added to the draw pile is: the card of the player who **did not** start the trick first, followed by the card of the player who **did** start the trick.
    * The identities of the cards in the draw pile are visible to both players.
    * The draw pile accumulates cards from consecutive draws.
    * The cards in the draw pile are contested in the next trick and awarded to the winner of that subsequent trick (along with the cards played in that trick).

7.  **Game End:**
    * The game ends immediately when one player runs out of cards (has zero cards in their deck). That player **loses**, and the other player **wins**.
    * If a maximum trick limit is reached (defined in `main.py`, default is high), the game ends, and the winner is determined by card count:
        * The player with more cards wins.
        * If both players have the same number of cards, the game is declared a draw.

## Code Structure

The project is organized into four main Python files in directory `top-trumps-ai/`:

1.  **`card_utils.py`:**
    * Contains utility functions for generating decks.
    * `id_generator()`: Creates unique random IDs for cards.
    * `generate_decks()`: Creates the full set of cards with stats based on specified parameters (number of cards, number of fields), ensures correct data types (Int64 for fields), shuffles the full deck, deals initial hands to Player 1 and Player 2, and returns the initial decks along with the list of field names and a full reference DataFrame (containing all generated cards and their stats).

2.  **`agents.py`:**
    * Defines the abstract base class `Agent`. All custom agents must inherit from this class.
    * Implements the base `Agent`'s `__init__` and `update_state` (which includes tracking the agent's own current card set `self.my_current_card_ids`).
    * Contains implementations for various built-in strategies:
        * `UserAgent`: Prompts a human player via the console. 
        * `RanderAgent`: Chooses a field randomly.
        * `MaxerAgent`: Always chooses the field with the highest raw value on its top card.
        * `RandMaxerAgent`: Probabilistically chooses between Maxer and Rander logic.
        * `MeanerMaxAgent`: Calculates normalization internally and chooses the field with the highest normalized value (z-score).
        * `EXPertAgent`: Tracks the *set* of cards the opponent might hold (via game history of the exchange of cards) and chooses the field maximizing expected score against that set. Uses optimized internal lookups.
        * `GoliathAgent`: Tracks *ordered possibilities* for the opponent's hand and chooses the field maximizing expected score against the possibilities for the top card. Uses optimized internal lookups and propagation of new knowledge throughout memory.
    * Includes the `create_agent` factory function to instantiate agents based on a strategy name string and provide them with necessary initialization data.

3.  **`game_engine.py`:**
    * Contains the `GameEngine` class, which orchestrates the game.
    * Initializes the game state using initial decks (extracting IDs into internal `deque` structures for efficiency) and the full reference deck (for lookups).
    * Manages the player `deque`s (holding card IDs) and the draw pile `deque`.
    * Enforces the game rules precisely as described above.
    * Calls agent methods (`choose_field`, `update_state`) providing only the allowed information via the defined API.
    * Uses an indexed internal copy of the reference deck for efficient card stat lookups when communicating with agents.

4.  **`main.py`:**
    * The main script to execute simulations.
    * `run_single_game()`: Sets up and runs one game between two specified agents, printing detailed trick information to the console. Useful for debugging or observing specific matchups.
    * `run_trials()`: Runs a batch of simulations based on configurations defined in a list (`trials_config`). Each configuration specifies the agents, game parameters (cards, fields), and number of trials. It aggregates win/loss/draw statistics.
    * `define_original_trials_config()`: A helper function providing a default list of agent matchups for `run_trials`.
    * Handles agent and engine creation, passing necessary initialization data (initial ID sets, reference deck).
    * Saves simulation results to `.json` and `.txt` files.

## Agent API & Information Constraints

To ensure fair testing and simulation, agents operate under strict information constraints. They interact with the `GameEngine` only through specific methods and receive limited information:

1.  **Initialization (`__init__`):**
    * When created by the `create_agent` factory, an agent's `__init__` receives:
        * `strategy_name` (string): Defined by the agent class.
        * `my_initial_card_ids` (Set[str]): A Python `set` containing the string IDs of all cards the agent starts the game with. **It does NOT receive the initial DataFrame, only the set of IDs, hiding the initial order.**
        * `full_game_deck_ref_df` (Optional[pd.DataFrame]): A pandas DataFrame containing *all* cards generated for the game, including their IDs and stats. Agents needing this (like Goliath, EXPert, MeanerMax) should store a copy (`self.full_game_deck_ref_df`) or use the indexed version created by the base class (`self._indexed_ref_df`). Simple agents can ignore this.
    * Agents needing to know the opponent's initial cards (e.g., Goliath, EXPert) **must derive** this information internally during `__init__` by subtracting `my_initial_card_ids` from the set of all IDs present in `full_game_deck_ref_df`.

2.  **Choosing a Move (`choose_field`):**
    * This method is called by the `GameEngine` **only** when it is the agent's turn to **start** a trick.
    * The method receives:
        * `my_top_card_df` (pd.DataFrame): A 1-row DataFrame containing the ID and all field stats for the single card currently at the top of the agent's deck.
        * `opponent_hand_size` (int): The number of cards the opponent currently has in their deck deque.
        * `available_fields` (List[str]): A list of the valid field names (strings) that can be chosen for the trick.
    * The agent **DOES NOT** receive its own full current deck (list or set) or any details about the opponent's cards beyond their deck size at this stage. It must rely on its internally tracked state if more information is needed for its strategy.

3.  **Updating State (`update_state`):**
    * This method is called by the `GameEngine` **after every trick** is resolved, for **both** players.
    * It provides detailed information about the outcome *from the perspective of the agent receiving the call*:
        * `outcome` (str): 'win', 'loss', or 'draw'.
        * `chosen_field` (str): The field that was competed on.
        * `my_card_played_df` (pd.DataFrame): 1-row DF of the card this agent just played.
        * `opponent_actual_card_id` (Optional[str]): The ID of the opponent's card (revealed on 'win' or 'draw' for the agent, `None` on 'loss').
        * `opponent_revealed_value` (Any): The value of the opponent's card on the `chosen_field` (always revealed).
        * `pot_cards_df` (pd.DataFrame): DF containing the two cards involved in a 'draw' outcome (empty otherwise).
        * `cards_i_won_df` (pd.DataFrame): DF containing cards this agent gained (opponent's played card + any pot cards). Empty if 'loss' or 'draw'.
        * `cards_opponent_won_df` (pd.DataFrame): DF containing cards the opponent gained (this agent's played card + any pot cards). Empty if 'win' or 'draw'.
    * **Agent Responsibility:** The agent **MUST** use this information to update its internal representation of the game state. At a minimum, it must track its own deck contents, or its actions will diverge from reality (tracked by the `GameEngine`). The base `Agent` class provides default logic in `update_state` to track `self.my_current_card_ids`. Agents overriding `update_state` (like Goliath, EXPert) must either call `super().update_state(...)` first or reimplement this self-tracking logic, in addition to updating their opponent models.

## Adding a New Agent

To create and test your own Top Trumps agent:

1.  **Create Agent Class:**
    * Open `agents.py`.
    * Define a new class that inherits from the base `Agent` class:
        ```python
        from agents import Agent # Make sure to import
        # Other necessary imports (pandas, numpy, etc.)

        class MyNewAgent(Agent):
            # ...
        ```

2.  **Implement `__init__`:**
    * Define the constructor. It MUST accept at least `my_initial_card_ids: Set[str]` and `full_game_deck_ref_df: Optional[pd.DataFrame]`.
    * Call the `super().__init__` method, passing your agent's unique strategy name (string), `my_initial_card_ids`, and `full_game_deck_ref_df`.
    * Store any necessary parameters (like `full_game_deck_ref_df` or the indexed `self._indexed_ref_df` from the base class) if your agent needs them.
    * Initialize any internal state variables your agent needs for tracking or decision-making (e.g., derive and store initial opponent possibilities).
        ```python
            def __init__(self,
                         my_initial_card_ids: Set[str],
                         full_game_deck_ref_df: Optional[pd.DataFrame],
                         # Add any other custom parameters your agent needs
                         my_custom_param: float = 0.1):

                super().__init__('MyNewStrat', my_initial_card_ids, full_game_deck_ref_df)
                self.my_custom_param = my_custom_param
                # Store indexed ref deck if needed for lookups
                self.ref_deck = self._indexed_ref_df # Use indexed from base
                if self.ref_deck is None and full_game_deck_ref_df is not None:
                     print("WARN MyNewAgent: Indexed ref deck failed, using non-indexed.")
                     self.ref_deck = self.full_game_deck_ref_df # Fallback

                # Example: Derive initial opponent state if needed
                # if self.ref_deck is not None:
                #    all_ids = set(self.ref_deck.index)
                #    opp_ids = all_ids - self.my_current_card_ids # my_current_card_ids holds initial set here
                #    self.internal_opponent_tracker = opp_ids.copy()
                # else:
                #    self.internal_opponent_tracker = set()
                # ... initialize other internal state ...
        ```

3.  **Implement `choose_field`:**
    * Define the method with the exact signature: `choose_field(self, my_top_card_df: pd.DataFrame, opponent_hand_size: int, available_fields: List[str]) -> str`.
    * Implement your strategy logic using ONLY the provided arguments and the agent's internal state (`self.my_current_card_ids`, `self.internal_opponent_tracker`, `self.ref_deck`, `self.my_custom_param`, etc.).
    * Return the chosen field name (string) from the `available_fields` list.
        ```python
            def choose_field(self, my_top_card_df, opponent_hand_size, available_fields) -> str:
                if not available_fields:
                    return "Field0" # Handle empty case

                # --- Your strategy logic here ---
                # Example: Choose randomly
                chosen_field = random.choice(available_fields)
                # --- End strategy logic ---

                return chosen_field
        ```

4.  **Implement `update_state`:**
    * Define the method with the exact signature matching the base class: `update_state(self, outcome: str, ..., cards_opponent_won_df: pd.DataFrame)`.
    * **Crucially: Call `super().update_state(...)` first** to ensure `self.my_current_card_ids` is correctly updated, unless you plan to reimplement that logic yourself. Pass all arguments received by your override to the `super()` call using the correct base class parameter names.
    * Add your agent's specific logic to update its internal state (e.g., opponent tracking) based on the trick outcome information provided in the arguments.
        ```python
            def update_state(self,
                             outcome: str, chosen_field: str, my_card_played_df: pd.DataFrame,
                             opponent_actual_card_id: Optional[str], opponent_revealed_value: Any,
                             pot_cards_df: pd.DataFrame, cards_i_won_df: pd.DataFrame,
                             cards_opponent_won_df: pd.DataFrame):

                # 1. Let the base class update self.my_current_card_ids
                super().update_state(outcome, chosen_field, my_card_played_df,
                                     opponent_actual_card_id, opponent_revealed_value,
                                     pot_cards_df, cards_i_won_df, cards_opponent_won_df)

                # 2. Add your agent's custom state update logic here
                # Example: Update internal_opponent_tracker based on outcome, IDs etc.
                # if outcome == 'loss' and not cards_opponent_won_df.empty:
                #     self.internal_opponent_tracker.update(set(cards_opponent_won_df['id']))
                # ... etc ...
        ```

5.  **Register in Factory:**
    * Open `agents.py` and find the `create_agent` factory function near the bottom.
    * Add an `elif` block for your new strategy name (lowercase).
    * Inside the block, check for any required custom parameters passed via the factory arguments (e.g., `my_custom_param`).
    * Instantiate your new agent class, passing the required base arguments (`my_initial_card_ids`, `full_game_deck_ref_df`) and any specific parameters.
        ```python
        # agents.py (Inside create_agent function)

            # ... other elif blocks ...
            elif strategy_lower == 'goliath':
                # ... goliath creation ...
                return GoliathAgent(...) # Use descriptive keywords

            # --- Add your new agent ---
            elif strategy_lower == 'mynewstrat':
                # Example: Check if a custom param was passed via factory args
                # custom_p = kwargs.get('my_custom_param_factory_name', 0.1) # Get with default
                # if full_game_deck_ref_df is None: # Check dependencies
                #     raise ValueError("MyNewStrat requires full_game_deck_ref_df")
                return MyNewAgent(**base_args) # Pass base args (ids, ref_df)
                                # Add custom args: my_custom_param=custom_p)

            else:
                raise ValueError(f"Unknown agent strategy provided: '{strategy}'")
        ```

6.  **Test:**
    * Open `main.py`.
    * Use your new strategy name (e.g., `'MyNewStrat'`) in the `run_single_game` function call or add a new configuration list to `run_trials` to test your agent against others. Remember to pass any custom parameters your agent needs via the `trials_config` list if running batch trials (matching the expected names in the factory).

## Running Simulations

Use the `main.py` script to run games:

1.  **Configure:**
    * **Single Game:** Modify the parameters within the `run_single_game(...)` call near the bottom of `main.py`. Set player strategies, card/field counts, debug flags, etc. Set `print_tricks=True` inside the `game_engine.play_game` call within `run_single_game` to see turn details.
    * **Batch Trials:** Modify the `define_original_trials_config()` function (or create your own list) to define the matchups, game parameters (`num_cards`, `num_trials`), and agent-specific parameters (like `randmax_fraction` or `EXPert_m_value`) for each set of trials. Modify the call to `run_trials(...)` to select the desired configurations and set global parameters like `number_of_fields` or `max_tricks_per_game`.
2.  **Run:** Execute the script from your terminal: `python main.py`
3.  **Results:**
    * Single game output is printed to the console.
    * Batch trial progress is printed to the console. Aggregate results are printed at the end. Results are also saved to timestamped `.json` (detailed) and `.txt` (summary) files in the same directory.

## Dependencies

* Python (>= 3.7 recommended for type hinting compatibility)
* pandas
* NumPy
* scikit-learn (for `StandardScaler` used by `MeanerMaxAgent`)

Install dependencies using pip:
`pip install -r requirements.txt`

## Agent performance benchmarking

The following table shows approximate Elo ratings for the implemented agents based on internal simulations under the following conditions:

* 50 cards per whole deck, i.e., 25 cards per player in initial split.
* 5 fields to choose from
* 10,000 games between each pair of agents
    * 5000 games each as Player 1 (there is a slight first-mover advantage)

The standard Elo distribution is fairly accurate for these Top Trumps agent, enabling estimating A-vs-C results from A-vs-B and B-vs-C. Therefore, these ratings provide a robust relative measure of strength. The `Rander` agent, making random choices, is set as the baseline with an Elo of 0. We would expect the Elo distribution to widen with the number of cards in the deck and with number of fields.

| Agent Strategy         | Approximate Elo Rating |
|------------------------|------------------------|
| Rander                 | 0                      |
| Maxer                  | 1180*                  |
| MeanerMax              | 1902                   |
| EXPert                 | 2009                   |
| Goliath                | 2411                   |

*Maxer vs Rander was too large a strength differential to benchmark with just games between the two, which prompted the development of Randmaxer.