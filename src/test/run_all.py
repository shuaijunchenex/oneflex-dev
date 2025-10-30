import experiment_entrency as experiment_entrency
import subprocess

from pathlib import Path
from typing import List

def list_yaml_files(folder: str, target_path: str) -> List[str]:
    """
    List all .yaml and .yml files under a folder (non-recursive),
    and return paths as target_path + filename.
    
    Args:
        folder (str): Source directory to scan.
        target_path (str): Prefix path for returned file names.
    
    Returns:
        List[str]: List of new paths (target_path + filename).
    """
    folder_path = Path(folder).resolve()
    if not folder_path.is_dir():
        raise NotADirectoryError(f"{folder} is not a valid directory")

    target = Path(target_path)
    files = []
    for p in sorted(list(folder_path.glob("*.yaml")) + list(folder_path.glob("*.yml"))):
        files.append(str(target / p.name))

    return files

BASE_DIR = Path(__file__).resolve().parent

import subprocess
import sys
import os
import time
from pathlib import Path
from datetime import datetime

def run_all(configs):
    child_code = (
        "import sys; "
        "import experiment_entrency; "
        "experiment_entrency.main(sys.argv[1])"
    )

    env = os.environ.copy()
    sep = ";" if os.name == "nt" else ":"
    env["PYTHONPATH"] = str(BASE_DIR) + (sep + env.get("PYTHONPATH", ""))

    successes = []  # [(cfg_path, seconds)]
    failures = []   # [(cfg_path, returncode or 'exception')]

    total = len(configs)
    for idx, cfg in enumerate(configs, 1):
        cfg_path = str(Path(cfg).resolve())
        print(f"\n[Batch] ({idx}/{total}) Running: {cfg_path}")
        cmd = [sys.executable, "-c", child_code, cfg_path]

        t0 = time.time()
        try:
            subprocess.run(cmd, check=True, cwd=str(BASE_DIR), env=env)
        except subprocess.CalledProcessError as e:
            elapsed = time.time() - t0
            print(f"[Batch][ERROR] {cfg_path} 执行失败，退出码 {e.returncode}，耗时 {elapsed:.2f}s（继续下一个）")
            failures.append((cfg_path, e.returncode))
            continue
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[Batch][ERROR] {cfg_path} 运行异常：{e}，耗时 {elapsed:.2f}s（继续下一个）")
            failures.append((cfg_path, "exception"))
            continue

        elapsed = time.time() - t0
        successes.append((cfg_path, elapsed))
        print(f"[Batch] Finished: {cfg_path} | {elapsed:.2f}s")

    # 汇总
    ok = len(successes)
    bad = len(failures)
    print("\n" + "="*60)
    print(f"[Batch][Summary] 总计 {total} | 成功 {ok} | 失败 {bad}")
    if successes:
        print("[成功样例] 前3条：")
        for p, s in successes[:3]:
            print(f"  - {p} ({s:.2f}s)")
    if failures:
        print("[失败列表]")
        for p, code in failures:
            print(f"  - {p} | 返回：{code}")

    # 写日志文件（在 BASE_DIR 下）
    try:
        log_path = Path(BASE_DIR) / "batch_summary.log"
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                    f"total={total} success={ok} fail={bad}\n")
            if successes:
                f.write("  successes:\n")
                for p, s in successes:
                    f.write(f"    - {p} ({s:.2f}s)\n")
            if failures:
                f.write("  failures:\n")
                for p, code in failures:
                    f.write(f"    - {p} (return={code})\n")
        print(f"[Batch] 结果已写入: {log_path}")
    except Exception as e:
        print(f"[Batch][WARN] 写入日志失败：{e}")


if __name__ == "__main__":
    configs = [
        "./fl_lora_sample/convergence_experiment/double_imbalance/sp_epoch1.yaml",
        "./fl_lora_sample/convergence_experiment/double_imbalance/rbla_epoch1.yaml",
    ]
    config_list = list_yaml_files("./fl_lora_sample/convergence_experiment/double_imbalance", 
                                  "./fl_lora_sample/convergence_experiment/double_imbalance")
    run_all(configs)


# if __name__ == "__main__":

#     config_list = list_yaml_files("./fl_lora_sample/convergence_experiment/", "./fl_lora_sample/convergence_experiment/")

#     configs = [
#         ("./fl_lora_sample/script_test-sp.yaml"),
#         ("./fl_lora_sample/script_test-rbla.yaml"),
#     ]

#     for i in config_list:
#         subprocess.run(experiment_entrency.main(i))
