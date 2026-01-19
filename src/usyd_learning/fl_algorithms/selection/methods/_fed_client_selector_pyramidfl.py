from __future__ import annotations

import dataclasses
import random
import numpy as np
import pandas as pd
from typing import Any, Dict, Iterable, List, Optional, Tuple

from ..fed_client_selector_args import FedClientSelectorArgs
from ..fed_client_selector_abc import FedClientSelector


# Client State Dataclass for PyramidFL
@dataclasses.dataclass
class _ClientState:
    """Client state tracking for PyramidFL selection
    
    Attributes:
        util_stat_ema: EWMA of statistical utility (loss reduction/accuracy gain)
        util_sys_ema: EWMA of system utility (throughput = 1 / time cost)
        avail_ema: EWMA of client availability (1 if observed, 0 otherwise)
        last_time: Most recent round time cost of the client
        last_seen_round: Last global round the client was observed
        participation_count: Number of times the client was selected (for exploration)
        stat_mean: Mean of historical statistical feedback (for exploration score)
        stat_std: Standard deviation of historical statistical feedback (for exploration score)
        stat_history: Historical record of statistical feedback values
    """
    util_stat_ema: float = 0.0
    util_sys_ema: float = 0.0
    avail_ema: float = 0.0
    last_time: float = 1.0
    last_seen_round: int = 0
    participation_count: int = 0
    stat_mean: float = 0.0
    stat_std: float = 0.0
    stat_history: List[float] = dataclasses.field(default_factory=list)


class FedClientSelector_PyramidFL(FedClientSelector):
    """
    PyramidFL client selection: line-level reconstruction of Algorithm 1 (server side).

    This class implements the server-side function SelectionAtServer(C, K, ε, T)
    in Algorithm 1 of PyramidFL paper (MobiCom 2022):
    
    Input:  
        - client set C, participant size K, exploitation factor ε
        - server pacer step Δ, straggler penalty α
        - update dropout bounds a,b
        
    Output: 
        - selected participants C_opt, statistical ranking R_stat
        - updated preferred duration T
    """

    def __init__(
        self,
        args: FedClientSelectorArgs|None = None,
        # Algorithm core hyperparameters (defaults)
        epsilon: float = 0.2,           # exploitation factor ε (∈ (0,1))
        delta_T: float = 0.1,           # server pacer step Δ for T update
        alpha_straggler: float = 1.0,   # straggler penalty coefficient α
        a_dropout: float = 0.5,         # lower bound a for B_i^+ (shard keep ratio)
        b_dropout: float = 1.0,         # upper bound b for B_i^+ (shard keep ratio)
        # EWMA coefficients (keep-old weights)
        stat_alpha: float = 0.7,        # EWMA weight for statistical utility
        sys_alpha: float = 0.7,         # EWMA weight for system utility
        avail_alpha: float = 0.8,       # EWMA weight for availability
        # Initial preferred round duration
        init_T: float = 1.0,
        min_time: float = 1e-3,         # Minimum time threshold to avoid division by zero
        # Client-side hyperparameters (for OptimizationAtClient)
        beta: float = 1.0,              # Confidence coefficient for adaptive iterations
        I_fix: int = 100,               # Fixed base iterations for local training
    ) -> None:
        super().__init__(args)

        # Exploit/Explore parameters
        self.epsilon = float(epsilon)
        self.delta_T = float(delta_T)
        self.alpha_straggler = float(alpha_straggler)

        # Dropout bounds for client-side shard keep ratio [a, b]
        self.a_dropout = float(a_dropout)
        self.b_dropout = float(b_dropout)

        # EWMA coefficients (keep-old weights)
        self.stat_alpha = float(stat_alpha)
        self.sys_alpha = float(sys_alpha)
        self.avail_alpha = float(avail_alpha)

        # Preferred round duration (global)
        self.T: float = float(init_T)
        self.min_time = float(min_time)

        # Client-side parameters for OptimizationAtClient
        self.beta = float(beta)
        self.I_fix = int(I_fix)

        # Per-client state storage
        self._state: Dict[str, _ClientState] = {}
        
        # Local parameters to distribute to selected clients
        self._client_local_params: Dict[str, Dict[str, Any]] = {}

        # Debug/introspection
        self._last_R_stat: List[str] = []   # Last statistical utility ranking
        
        # Random seed initialized by base class call to with_random_seed if args present
        # but we also have local rng
        self._rng = random.Random(2024)
        if args:
            self.with_random_seed(args.random_seed)

    def with_random_seed(self, seed: int = -1):
        super().with_random_seed(seed)
        # Also seed our local RNG
        # Note: super().with_random_seed uses 'random' module global seed.
        if seed > 0:
            self._rng = random.Random(seed)
        return self

    def _flatten_data(self, d: dict, parent_key: str = '', sep: str = '_') -> dict:
        """Helper to flatten nested dictionary from client data"""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_data(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
                if k not in d: 
                    items.append((k, v))
        return dict(items)

    def select(self, client_list: list, select_number: int = -1):
        """
        Main entry point for framework selection
        """
        if select_number <= 0:
            select_number = self.select_number # Use property from base class

        # Prepare metric matrix by flattening client data
        metric_matrix = {}
        for cid, data in self._clients_data_dict.items():
            if isinstance(data, dict):
                metric_matrix[cid] = self._flatten_data(data)
            else:
                metric_matrix[cid] = data

        name_dict = {str(c.node_id): str(c.name) for c in client_list}
        
        selected_ids = self.select_clients_internal(
            metric_matrix=metric_matrix,
            name_dict=name_dict,
            number=select_number,
            current_round=self.select_round
        )
        
        return [c for c in client_list if str(c.node_id) in selected_ids]

    def select_clients_internal(
        self,
        metric_matrix: Any,
        name_dict: Dict[str, str],
        number: int,
        break_client_set: Optional[Iterable[str]] = None,
        current_round: int = 0
    ) -> List[str]:
        """
        Server-side client selection (Algorithm 1: SelectionAtServer).
        """
        if number <= 0:
            self._print_selected_clients([], name_dict)
            return []
        
        # Process break client set
        break_set = {str(x) for x in (break_client_set or [])}

        # 0. Normalize metrics to DataFrame and filter invalid clients
        df = self._to_dataframe(metric_matrix, name_dict)
        if df.empty:
            self._print_selected_clients([], name_dict)
            return []

        # Ensure client_id column exists and filter broken clients
        if "client_id" not in df.columns:
            df["client_id"] = list(name_dict.keys())[: len(df)]
        df["client_id"] = df["client_id"].astype(str)
        df = df[~df["client_id"].isin(break_set)].copy()
        if df.empty:
            self._print_selected_clients([], name_dict)
            return []

        # C: All available clients at server
        C: List[str] = df["client_id"].tolist()

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 4–5: GetClientFeedback()
        # ----------------------------------------------------------------------
        F_stat, F_sys, sample_num_map, shard_keep_ratio_map = self._get_client_feedback(df)

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 6: UpdateClient(C, F_stat, F_sys)
        # ----------------------------------------------------------------------
        C_E, Util_stat, t_map = self._update_client(C, F_stat, F_sys, current_round)

        if not C_E:
            self._print_selected_clients([], name_dict)
            return []

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 7: UpdatePreferDuration(F_stat, T, Δ)
        # ----------------------------------------------------------------------
        self.T = self._update_preferred_duration(F_stat, F_sys, self.T, self.delta_T)

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 8–9: Calculate global utility for each client
        # ----------------------------------------------------------------------
        Util = self._compute_global_utility(
            C_E, Util_stat, t_map, sample_num_map, shard_keep_ratio_map
        )

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 10: SelectForExploit(C_E, Util, εK)
        # ----------------------------------------------------------------------
        K = min(number, len(C_E))
        k_exploit = int(max(1, round(self.epsilon * K)))
        C_star = self._select_for_exploit(C_E, Util, k_exploit)

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 11: SelectForExplore(C_E, Util_sys, (1−ε)K)
        # ----------------------------------------------------------------------
        Util_sys = {cid: self._state[cid].util_sys_ema for cid in C_E}
        k_explore = K - len(C_star)
        C_opt = self._select_for_explore(C_E, C_star, Util_sys, k_explore)

        # ----------------------------------------------------------------------
        # Algorithm 1 Line 12: RankingClients(F_stat)
        # ----------------------------------------------------------------------
        self._last_R_stat = self._ranking_clients(F_stat)

        # ----------------------------------------------------------------------
        # Generate client-side parameters (Algorithm 1 Lines 14-17)
        # ----------------------------------------------------------------------
        self._generate_client_local_params(C_opt, t_map)

        # Print selection result and return
        self._print_selected_clients(C_opt, name_dict)
        return C_opt

    def _get_client_feedback(
        self, df: pd.DataFrame
    ) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]:
        """
        Algorithm 1: GetClientFeedback()
        """
        F_stat: Dict[str, float] = {}
        F_sys: Dict[str, float] = {}
        sample_num_map: Dict[str, float] = {}
        shard_keep_ratio_map: Dict[str, float] = {}

        for _, row in df.iterrows():
            cid = str(row["client_id"])

            # Extract system feedback (t_i: round time cost)
            t_i = self._extract_time(row)
            F_sys[cid] = t_i

            # Extract statistical feedback
            loss_before = float(
                row.get("prev_loss", row.get("loss_before", row.get("train_loss_before", 0.0))) or 0.0
            )
            loss_after = float(
                row.get("batch_loss", row.get("loss_after", row.get("loss", loss_before))) or loss_before
            )
            delta_loss = max(0.0, loss_before - loss_after)

            acc = float(row.get("val_acc", row.get("client_val_acc", 0.0)) or 0.0)
            acc_prev = float(row.get("prev_val_acc", row.get("client_prev_val_acc", 0.0)) or 0.0)
            delta_acc = max(0.0, acc - acc_prev)

            F_stat[cid] = max(0.0, delta_loss + delta_acc)

            # Extract sample count (|B_i|)
            sample_num_map[cid] = float(row.get("sample_num", row.get("n_data", 1.0)))

            # Extract shard keep ratio (B_i^+)
            shard_keep_ratio_map[cid] = float(row.get("shard_keep_ratio", 1.0))

        return F_stat, F_sys, sample_num_map, shard_keep_ratio_map

    def _update_client(
        self,
        C: List[str],
        F_stat: Dict[str, float],
        F_sys: Dict[str, float],
        current_round: int,
    ) -> Tuple[List[str], Dict[str, float], Dict[str, float]]:
        """
        Algorithm 1: UpdateClient(C, F_stat, F_sys)
        """
        C_E: List[str] = []
        Util_stat: Dict[str, float] = {}
        t_map: Dict[str, float] = {}

        for cid in C:
            if cid not in F_sys:
                continue

            t_i = max(self.min_time, float(F_sys.get(cid, 1.0)))
            s_i = float(F_stat.get(cid, 0.0))

            state = self._state.setdefault(cid, _ClientState())

            state.util_stat_ema = (
                self.stat_alpha * state.util_stat_ema
                + (1.0 - self.stat_alpha) * s_i
            )

            state.stat_history.append(s_i)
            if len(state.stat_history) > 10:
                state.stat_history.pop(0)

            if state.stat_history:
                state.stat_mean = np.mean(state.stat_history)
                state.stat_std = np.std(state.stat_history)
            else:
                state.stat_mean = 0.0
                state.stat_std = 0.0

            sys_util = 1.0 / t_i
            state.util_sys_ema = (
                self.sys_alpha * state.util_sys_ema
                + (1.0 - self.sys_alpha) * sys_util
            )

            state.avail_ema = (
                self.avail_alpha * state.avail_ema
                + (1.0 - self.avail_alpha) * 1.0
            )

            state.last_time = t_i
            state.last_seen_round = current_round

            C_E.append(cid)
            Util_stat[cid] = state.util_stat_ema
            t_map[cid] = t_i

        return C_E, Util_stat, t_map

    def _update_preferred_duration(
        self,
        F_stat: Dict[str, float],
        F_sys: Dict[str, float],
        T: float,
        delta_T: float,
    ) -> float:
        """
        Algorithm 1: UpdatePreferDuration(F_stat, T, Δ)
        """
        if not F_sys:
            return T

        times = np.array(list(F_sys.values()), dtype=float)
        median_t = float(np.median(times))

        if not F_stat:
            stat_factor = 1.0
        else:
            stat_mean = np.mean(list(F_stat.values()))
            stat_factor = max(0.5, 2.0 - stat_mean)

        new_T = (1.0 - delta_T) * T + delta_T * median_t * stat_factor
        return max(self.min_time, new_T)

    def _compute_global_utility(
        self,
        C_E: List[str],
        Util_stat: Dict[str, float],
        t_map: Dict[str, float],
        sample_num_map: Dict[str, float],
        shard_keep_ratio_map: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Algorithm 1 Lines 8–9: Calculate global utility for each client
        """
        Util: Dict[str, float] = {}

        for cid in C_E:
            state = self._state.setdefault(cid, _ClientState())
            t_i = max(self.min_time, t_map.get(cid, state.last_time or 1.0))
            
            n_samples = sample_num_map.get(cid, 1.0)
            B_plus = shard_keep_ratio_map.get(cid, 1.0)
            B_plus = max(self.a_dropout, min(self.b_dropout, B_plus))
            
            util_stat_i = Util_stat.get(cid, state.util_stat_ema)
            
            if t_i > self.T:
                util_i = 0.0
            else:
                util_i = n_samples * util_stat_i * (self.T / t_i) * self.alpha_straggler
            
            Util[cid] = max(0.0, B_plus * util_i)

        return Util

    def _select_for_exploit(
        self,
        C_E: List[str],
        Util: Dict[str, float],
        k_exploit: int,
    ) -> List[str]:
        """
        Algorithm 1: SelectForExploit(C_E, Util, εK)
        """
        if not C_E:
            return []
        
        k_exploit = max(1, min(k_exploit, len(C_E)))
        
        sorted_by_util = sorted(
            C_E, key=lambda cid: Util.get(cid, 0.0), reverse=True
        )
        selected = sorted_by_util[:k_exploit]
        
        for cid in selected:
            self._state[cid].participation_count += 1

        return selected

    def _select_for_explore(
        self,
        C_E: List[str],
        C_star: List[str],
        Util_sys: Dict[str, float],
        k_explore: int,
    ) -> List[str]:
        """
        Algorithm 1: SelectForExplore(C_E, Util_sys, (1−ε)K)
        """
        if k_explore <= 0:
            return C_star

        remaining = [cid for cid in C_E if cid not in C_star]
        if not remaining:
            return C_star

        k_explore = min(k_explore, len(remaining))

        def _explore_score(cid: str) -> float:
            state = self._state[cid]
            participation_penalty = 1.0 / (state.participation_count + 1)
            
            if len(state.stat_history) < 2:
                stat_cv = 1.0
            else:
                stat_cv = state.stat_std / (state.stat_mean + 1e-6)
            
            sys_util = Util_sys.get(cid, 0.0)
            return participation_penalty * stat_cv * sys_util

        remaining_sorted = sorted(
            remaining, key=_explore_score, reverse=True
        )
        C_explore = remaining_sorted[:k_explore]

        for cid in C_explore:
            self._state[cid].participation_count += 1

        return C_star + C_explore

    def _ranking_clients(self, F_stat: Dict[str, float]) -> List[str]:
        """
        Algorithm 1: RankingClients(F_stat)
        """
        return sorted(F_stat.keys(), key=lambda cid: F_stat[cid], reverse=True)

    def _generate_client_local_params(self, C_opt: List[str], t_map: Dict[str, float]) -> None:
        """
        Generate client-side parameters for OptimizationAtClient (Algorithm 1 Lines 14-17)
        """
        client_local_params = {}
        for cid in C_opt:
            r_stat = self._last_R_stat.index(cid) if cid in self._last_R_stat else len(self._last_R_stat)
            P_i = self.a_dropout + (self.b_dropout - self.a_dropout) / len(C_opt) * r_stat
            
            t_i = t_map.get(cid, 1.0)
            t_i_comp = t_i * 0.8
            
            I_i = (self.beta * max(0.0, self.T - t_i) / (t_i_comp + 1e-6) + 1) * self.I_fix
            
            client_local_params[cid] = {
                "shard_keep_ratio": P_i,
                "adaptive_iter": int(I_i)
            }
        
        self._client_local_params = client_local_params

    def _extract_time(self, row: pd.Series) -> float:
        """
        Extract wall-clock time cost from metric row.
        """
        time_keys = [
            "time_cost", "total_time_cost", "round_time",
            "training_time", "train_time"
        ]
        for key in time_keys:
            if key in row and row[key] is not None:
                try:
                    return max(self.min_time, float(row[key]))
                except (TypeError, ValueError):
                    continue
        return self.min_time

    def _to_dataframe(self, metric_matrix: Any, name_dict: Dict[str, str]) -> pd.DataFrame:
        """
        Normalize various metric matrix formats to a pandas DataFrame.
        """
        if isinstance(metric_matrix, pd.DataFrame):
            df = metric_matrix.copy()
        elif isinstance(metric_matrix, list):
            if not metric_matrix:
                df = pd.DataFrame()
            elif isinstance(metric_matrix[0], dict):
                df = pd.DataFrame(metric_matrix)
            else:
                df = pd.DataFrame(metric_matrix)
        elif isinstance(metric_matrix, dict):
            records: List[Dict[str, Any]] = []
            for cid, metrics in metric_matrix.items():
                if isinstance(metrics, dict):
                    rec = dict(metrics)
                    rec["client_id"] = cid
                else:
                    rec = {"client_id": cid, "value": metrics}
                records.append(rec)
            df = pd.DataFrame(records)
        else:
            df = pd.DataFrame()

        if "client_id" not in df.columns and name_dict:
            df["client_id"] = list(name_dict.keys())[: len(df)]
        
        return df

    def _print_selected_clients(self, selected: List[str], name_dict: Dict[str, str]) -> None:
        """
        Print selected clients (debug/introspection).
        """
        pass # Suppressed output to avoid console spam

