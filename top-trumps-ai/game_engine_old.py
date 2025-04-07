# game_engine.py
"""
GameEngine providing minimal info to agents: top card, opp size, fields.
"""
import random
from typing import Optional, Tuple, Dict, Any, Set
import numpy as np
import pandas as pd
from agents import Agent

class GameEngine:
    """Manages game flow, provides minimal info to agents."""
    def __init__(self, player1_agent: Agent, player2_agent: Agent, initial_deck1: pd.DataFrame, initial_deck2: pd.DataFrame, field_names: list, engine_debug: bool = False):
        if not isinstance(player1_agent, Agent) or not isinstance(player2_agent, Agent): raise TypeError("Agents must be instances of Agent")
        if not isinstance(initial_deck1, pd.DataFrame) or not isinstance(initial_deck2, pd.DataFrame): raise TypeError("Initial decks must be DataFrames")
        if 'id' not in initial_deck1.columns or 'id' not in initial_deck2.columns: raise ValueError("Decks need 'id' column")
        if not isinstance(field_names, list) or not field_names: raise ValueError("field_names must be non-empty list")
        for field in field_names:
            if field not in initial_deck1.columns or field not in initial_deck2.columns: raise ValueError(f"Field '{field}' not in decks.")
        self.player1_agent=player1_agent; self.player2_agent=player2_agent
        self._initial_deck1=initial_deck1.copy(); self._initial_deck2=initial_deck2.copy()
        self.field_colnames=list(field_names); self.n_cards_total=len(initial_deck1)+len(initial_deck2)
        self.n_fields=len(field_names); self._min_compare_val=-999999; self._debug=engine_debug
        self.player1_deck: pd.DataFrame=pd.DataFrame(); self.player2_deck: pd.DataFrame=pd.DataFrame()
        self.draw_pile: pd.DataFrame=pd.DataFrame(); self.trick_starter_index: int=0
        self.last_trick_outcome_for_p1: Optional[str]=None; self.trick_count: int=0

    def setup_game(self):
        self._log_engine("Setting up game..."); self.player1_deck=self._initial_deck1.copy(); self.player2_deck=self._initial_deck2.copy()
        self.draw_pile=pd.DataFrame(columns=self._initial_deck1.columns); self.trick_starter_index=0
        self.last_trick_outcome_for_p1=None; self.trick_count=0; self._log_engine(f"Setup done. P1:{len(self.player1_deck)}, P2:{len(self.player2_deck)}")

    def _log_engine(self, *args):
        if self._debug: print(f"ENGINE DEBUG (Trick {self.trick_count}):", *args)

    def _get_game_state_for_agent(self, agent_index: int) -> Optional[Dict[str, Any]]:
        """ Prepares state view dict - NO current deck IDs passed."""
        my_deck = self.player1_deck if agent_index == 0 else self.player2_deck
        opponent_deck = self.player2_deck if agent_index == 0 else self.player1_deck
        if my_deck.empty: self._log_engine(f"Agent {agent_index+1} empty."); return None
        agent_state_view = {
            "my_top_card_df": my_deck.iloc[[0]].copy(),
            "opponent_hand_size": len(opponent_deck),
            "available_fields": self.field_colnames,
        }
        return agent_state_view

    def play_trick(self, print_trick_details: bool = True) -> Optional[str]:
        """Plays a single trick."""
        self.trick_count+=1; self._log_engine(f"--- Trick {self.trick_count} ---")
        game_over=self.is_game_over();
        if game_over: self._log_engine(f"Game over pre-trick ({game_over})."); self.trick_count-=1; return game_over
        idx=self.trick_starter_index; agent=self.player1_agent if idx==0 else self.player2_agent
        s_name=f"P{idx+1}({agent.strategy_name})"; self._log_engine(f"Starter: {s_name}")
        state_view=self._get_game_state_for_agent(idx)
        if state_view is None: self._log_engine("Starter empty."); self.trick_count-=1; return self.is_game_over()
        chosen_field=None
        try:
            chosen_field = agent.choose_field(**state_view) # Pass dict directly
            if chosen_field not in self.field_colnames: print(f"Warn: Agent {agent.strategy_name} invalid field '{chosen_field}'. Random."); chosen_field = random.choice(self.field_colnames)
        except Exception as e: print(f"CRIT Err agent choose_field: {e}. Random."); chosen_field = random.choice(self.field_colnames)
        self._log_engine(f"{s_name} chose: {chosen_field}")
        if self.player1_deck.empty or self.player2_deck.empty: self._log_engine("Deck empty?"); self.trick_count-=1; return self.is_game_over()
        p1_c_df=self.player1_deck.iloc[[0]]; p2_c_df=self.player2_deck.iloc[[0]]
        p1_id=p1_c_df['id'].iloc[0]; p2_id=p2_c_df['id'].iloc[0]; self._log_engine(f"P1:'{p1_id}', P2:'{p2_id}'")
        p1_v=self._get_card_value(p1_c_df, chosen_field); p2_v=self._get_card_value(p2_c_df, chosen_field); self._log_engine(f"Compare '{chosen_field}': P1={p1_v}, P2={p2_v}")
        if print_trick_details: print(f"-\nTrick {self.trick_count}: {s_name} chose {chosen_field}\n P1({self.player1_agent.strategy_name}): {p1_id} ({chosen_field}={p1_v})\n P2({self.player2_agent.strategy_name}): {p2_id} ({chosen_field}={p2_v})")
        self.player1_deck=self.player1_deck.iloc[1:].reset_index(drop=True); self.player2_deck=self.player2_deck.iloc[1:].reset_index(drop=True)
        outcome_p1:str='draw'; add_p1=[]; add_p2=[]; pot_upd=pd.DataFrame(columns=self._initial_deck1.columns)
        if p1_v > p2_v:
            outcome_p1='win'; self._log_engine("P1 Wins"); print(" Res: P1 wins.") if print_trick_details else None; add_p1=[p1_c_df,p2_c_df];
            if not self.draw_pile.empty: self._log_engine(f" P1 wins draw ({len(self.draw_pile)})"); add_p1.append(self.draw_pile); self.draw_pile=pd.DataFrame(columns=self._initial_deck1.columns)
            self.player1_deck=pd.concat([self.player1_deck]+add_p1, ignore_index=True);
            for col in self.field_colnames: self.player1_deck[col]=pd.to_numeric(self.player1_deck[col],errors='coerce').astype('Int64')
        elif p2_v > p1_v:
            outcome_p1='loss'; self._log_engine("P2 Wins"); print(" Res: P2 wins.") if print_trick_details else None; add_p2=[p2_c_df,p1_c_df];
            if not self.draw_pile.empty: self._log_engine(f" P2 wins draw ({len(self.draw_pile)})"); add_p2.append(self.draw_pile); self.draw_pile=pd.DataFrame(columns=self._initial_deck1.columns)
            self.player2_deck=pd.concat([self.player2_deck]+add_p2, ignore_index=True);
            for col in self.field_colnames: self.player2_deck[col]=pd.to_numeric(self.player2_deck[col],errors='coerce').astype('Int64')
        else:
            outcome_p1='draw'; self._log_engine("Draw"); print(" Res: Draw.") if print_trick_details else None; draw_add=[p2_c_df,p1_c_df] if idx==0 else [p1_c_df,p2_c_df]; self.draw_pile=pd.concat([self.draw_pile]+draw_add, ignore_index=True); self._log_engine(f" Added {len(draw_add)} pile. Tot:{len(self.draw_pile)}")
            if not self.draw_pile.empty:
                for col in self.field_colnames: self.draw_pile[col]=pd.to_numeric(self.draw_pile[col],errors='coerce').astype('Int64')
            pot_upd=pd.concat(draw_add, ignore_index=True)
        p1_won=pd.concat(add_p1[1:],ignore_index=True) if len(add_p1)>1 else pd.DataFrame(columns=p1_c_df.columns)
        p2_won=pd.concat(add_p2[1:],ignore_index=True) if len(add_p2)>1 else pd.DataFrame(columns=p2_c_df.columns)
        p1_info={"outcome":outcome_p1,"chosen_field":chosen_field,"my_card_played_df":p1_c_df,"opponent_actual_card_id":p2_id if outcome_p1 in ['win','draw'] else None,"opponent_revealed_value":p2_v,"pot_cards_df":pot_upd,"cards_i_won_df":p1_won,"cards_opponent_won_df":p2_won}
        p2_outcome={'win':'loss','loss':'win','draw':'draw'}.get(outcome_p1)
        p2_info={"outcome":p2_outcome,"chosen_field":chosen_field,"my_card_played_df":p2_c_df,"opponent_actual_card_id":p1_id if p2_outcome in ['win','draw'] else None,"opponent_revealed_value":p1_v,"pot_cards_df":pot_upd,"cards_i_won_df":p2_won,"cards_opponent_won_df":p1_won}
        self._log_engine("Notifying...");
        try: self.player1_agent.update_state(**p1_info)
        except Exception as e: print(f"CRIT Err P1 update: {e}")
        try: self.player2_agent.update_state(**p2_info)
        except Exception as e: print(f"CRIT Err P2 update: {e}")
        self.last_trick_outcome_for_p1=outcome_p1
        if outcome_p1=='win': self.trick_starter_index=0
        elif outcome_p1=='loss': self.trick_starter_index=1
        self._log_engine(f"Trick end. P1:{len(self.player1_deck)}, P2:{len(self.player2_deck)}, Draw:{len(self.draw_pile)}. Next: {self.trick_starter_index}")
        return self.is_game_over()

    def _get_card_value(self, card_df: pd.DataFrame, field: str) -> int:
        try: v=card_df[field].iloc[0]; return int(v) if pd.notna(v) else self._min_compare_val
        except Exception: return self._min_compare_val
    def is_game_over(self) -> Optional[str]:
        p1c=len(self.player1_deck); p2c=len(self.player2_deck); total=p1c+p2c+len(self.draw_pile)
        if total!=self.n_cards_total: self._log_engine(f"CRIT WARN: Count mismatch! Exp {self.n_cards_total}, Found {total}")
        if p1c==0 and p2c==0: return 'player2'
        if p1c==0: return 'player2'
        if p2c==0: return 'player1'
        return None
    def play_game(self, print_tricks: bool = False, max_tricks: int = 5000) -> str:
        self.setup_game()
        while self.trick_count < max_tricks:
            winner = self.play_trick(print_trick_details=print_tricks)
            if winner:
                if print_tricks: print(f"===\nGAME OVER @ {self.trick_count}. Win: {winner.upper()}\n P1:{len(self.player1_deck)}, P2:{len(self.player2_deck)}\n===")
                return winner
        print(f"\nWarn: Max tricks ({max_tricks})."); p1c=len(self.player1_deck); p2c=len(self.player2_deck); print(f" Counts: P1={p1c}, P2={p2c}, Draw={len(self.draw_pile)}")
        if p1c>p2c: return 'player1'
        elif p2c>p1c: return 'player2'
        else: return 'draw'