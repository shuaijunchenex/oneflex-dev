from __future__ import annotations

import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Set

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


class FedClientSelector_FedGRA(FedClientSelector):
    """
    FedGRA client selection algorithm using Grey Relational Analysis.
    """

    def __init__(self, args: FedClientSelectorArgs|None = None):
        super().__init__(args)
        # Track staleness (rounds since last participation) for fairness
        # Key: client_id, Value: staleness count
        self._staleness: Dict[str, int] = {}
        
        # Load args
        self.fedgra_mode = self._args.get("fedgra_mode", "all_participate")
        self.min_participate_round = self._args.get("min_participate_round", 10) # interpreted as 'staleness threshold'

    def select(self, client_list: list, select_number: int = -1):
        """
        Main framework entry point
        """
        if select_number <= 0:
            select_number = self.select_number

        # 1. Update staleness for all available clients
        # Increment staleness for everyone present in this round's candidate list
        current_cids = [str(c.node_id) for c in client_list]
        for cid in current_cids:
            self._staleness[cid] = self._staleness.get(cid, 0) + 1

        # 2. Build GRG Matrix & Fairness Matrix
        # Default strategy: Use 'grg' metric if available, else 'loss', else 0.0
        grg_matrix = {}
        
        # We need data for the current clients. 
        # _clients_data_dict comes from the previous round usually.
        for cid in current_cids:
            data = self._clients_data_dict.get(cid, {})
            val = 0.0
            if isinstance(data, dict):
                # Try getting explicit GRG score, or fallback to record checks
                if "grg" in data:
                    val = float(data["grg"])
                elif "fedgra_score" in data:
                    val = float(data["fedgra_score"])
                elif "train_record" in data and isinstance(data["train_record"], dict):
                    # Fallback to loss (assuming high loss = high priority)
                    val = float(data["train_record"].get("loss", 0.0))
                else:
                    val = float(data.get("loss", 0.0))
            else:
                # If data is just a value?
                try: 
                    val = float(data)
                except: 
                    val = 0.0
            
            grg_matrix[cid] = val

        # Fairness matrix is actually the staleness map
        fairness_matrix = {cid: self._staleness.get(cid, 0) for cid in current_cids}

        name_dict = {str(c.node_id): str(c.name) for c in client_list}

        # 3. Call the core algorithm
        selected_ids = self.select_clients(
            grg_matrix=grg_matrix,
            fairness_matrix=fairness_matrix,
            name_dict=name_dict,
            number=select_number,
            fedgra_mode=self.fedgra_mode,
            min_participate_round=self.min_participate_round
        )

        # 4. Post-selection: Reset staleness for selected clients
        for cid in selected_ids:
            self._staleness[cid] = 0

        # Return objects
        return [c for c in client_list if str(c.node_id) in selected_ids]

    def select_clients(self, grg_matrix, fairness_matrix, name_dict, number, 
                      fedgra_mode='all_participate', min_participate_round=10, 
                      break_client_set=None, **kwargs):
        """
        Select clients using FedGRA algorithm.
        
        Args:
            grg_matrix: Grey relational grade matrix (Client ID -> GRG Score)
            fairness_matrix: Client fairness scores (Client ID -> Staleness/Rounds)
            name_dict: Dictionary mapping client IDs to names
            number: Number of clients to select
            fedgra_mode: Selection mode ('used' or 'all_participate')
            min_participate_round: Minimum staleness/rounds for fairness selection
            break_client_set: Set of broken clients to exclude
            **kwargs: Additional arguments (ignored)
            
        Returns:
            List of selected client IDs
        """
        if break_client_set is None:
            break_client_set = set()
        else:
            break_client_set = set(break_client_set)
            
        if fedgra_mode == 'used':
            return self._vanilla_selection(grg_matrix, name_dict, number)
        elif fedgra_mode == 'all_participate':
            return self._all_participate_selection(
                grg_matrix, fairness_matrix, name_dict, number, 
                min_participate_round, break_client_set
            )
    
    def _vanilla_selection(self, grg_matrix, name_dict, number):
        """Vanilla FedGRA selection method."""
        # Determine number of clients to pick
        if number < 1:
            clients_to_pick = max(1, round(number * len(grg_matrix)))
        else:
            clients_to_pick = int(number)
        
        # Convert dict to dataframe and rank by GRG
        # print(type(grg_matrix))
        if not grg_matrix:
            return []
            
        df = pd.DataFrame.from_dict(grg_matrix, orient='index', columns=['Value'])
        client_rank = df.sort_values(ascending=False, by='Value')
        
        # Get selected clients
        selected_client = client_rank[:clients_to_pick].index.tolist()
        # keys = [value for key, value in name_dict.items() if key in selected_client]
        # print(keys)
        
        self._print_selected_clients(selected_client, name_dict)
        return selected_client
    
    def _all_participate_selection(self, grg_matrix, fairness_matrix, name_dict, 
                                 number, min_participate_round, break_client_set):
        """All participate selection method with fairness consideration."""
        # Determine number of clients to pick
        if number < 1:
            clients_to_pick = max(1, round(number * len(grg_matrix)))
        else:
            clients_to_pick = int(number)
        
        # Select clients by fairness (participation rounds)
        fairness_clients = [
            key for key, value in fairness_matrix.items() 
            if value >= min_participate_round
        ]

        # Use set intersection to filter
        fairness_clients_set = {
            key for key in fairness_clients 
            if key not in break_client_set
        }
        
        # Note: Code logic says "If enough... return them". 
        # This implies we ONLY return fairness clients if we have enough.
        # Otherwise we pad with GRG.
        
        if len(fairness_clients_set) >= number:
            # We have more fairness candidates than needed? 
            # Or exactly enough?
            # The original code just returns list(fairness_clients).
            # This might return MORE than `number`. Assume this is intentional logic from user.
            keys = [name_dict.get(key, key) for key in fairness_clients_set]
            # print("Clients selected by fairness:", keys)
            self._print_selected_clients(list(fairness_clients_set), name_dict, prefix="Fairness Selection")
            return list(fairness_clients_set)
        
        # Select remaining clients by GRG
        # Filter out those already selected via fairness
        clients_left = {
            key: value for key, value in grg_matrix.items() 
            if key not in fairness_clients_set and key not in break_client_set
        }
        
        if not clients_left and not fairness_clients_set:
            return []

        # Select additional clients needed
        num_additional = clients_to_pick - len(fairness_clients_set)
        
        if num_additional > 0 and clients_left:
            df = pd.DataFrame.from_dict(clients_left, orient='index', columns=['Value'])
            client_rank = df.sort_values(ascending=False, by='Value')
            additional_clients = client_rank[:num_additional].index.tolist()
        else:
            additional_clients = []
        
        # Combine fairness and GRG-selected clients
        selected_client = list(fairness_clients_set) + additional_clients
        
        # Print results
        self._print_selected_clients(selected_client, name_dict)
        
        return selected_client

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, str], prefix: str = "FedGRA"):
        """Helper to print selected clients"""
        pass
        # names = [name_dict.get(cid, cid) for cid in selected]
        # print(f"{prefix}: Selected {len(selected)} - {names}")
