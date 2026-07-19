# Copyright(C) [2026] Advanced Micro Devices, Inc. All rights reserved.
# utils/registry.py

from pathlib import Path
from typing import Dict, Optional


def is_flydsl_rewrite(task_type: str) -> bool:
    """True for cross-language rewrite-to-FlyDSL tasks (``<src>2flydsl``, src != flydsl).

    A rewrite task has a DIFFERENT source and target language, so the agent must
    re-implement the kernel in the target language rather than optimize the source
    in place. These are driven by KernelForge's ``forge-rewrite-by-flydsl`` layer
    (rewrite the source into FlyDSL, then reuse forge-loop to optimize it).

    ``flydsl2flydsl`` is NOT a rewrite (source == target): it optimizes an existing
    FlyDSL kernel and uses the generic/forge-loop path. The target language for
    forge-rewrite is always FlyDSL today, so only ``*2flydsl`` (src != flydsl)
    routes here; other cross-language pairs (e.g. ``torch2hip``) have their own
    handling.
    """
    tt = (task_type or "").strip().lower()
    if "2" not in tt:
        return False
    src, _, dst = tt.partition("2")
    return dst == "flydsl" and bool(src) and src != "flydsl"


def get_task_config(tasks_root: str = "tasks", category: Optional[str] = None) -> Dict[str, str]:
    """
    Automatically scan all task folders under the tasks directory.
    If config.yaml exists, register it to task_config_list.

    Task naming strategy:
    - Uses relative path from tasks_root as the unique task name
    - Example: tasks/customer_hip/mmcv/ball_query/config.yaml
      -> task_name: "customer_hip/mmcv/ball_query"
    - This ensures uniqueness even when folder names collide

    Args:
        tasks_root: Root directory for tasks (default: "tasks")
        category: Optional category to filter tasks (e.g., "customer_hip", "rocm-examples")

    Returns:
        dict: {task_name: config_path}
            - task_name: Relative path from tasks_root to the task folder
            - config_path: Relative path to config.yaml from project root
    """
    task_config_dict = {}
    root = Path(tasks_root)

    # If category is specified, only scan that category
    if category:
        pattern = f"{category}/**/config.yaml"
    else:
        pattern = "**/config.yaml"

    for config_path in root.glob(pattern):
        # Get task name as relative path from tasks_root to the task folder
        # This ensures uniqueness across different categories
        # Example: customer_hip/mmcv/ball_query
        task_folder = config_path.parent
        task_name = str(task_folder.relative_to(root))

        # Store relative path from project root to config.yaml
        relative_path = str(config_path)

        # Register to task_config_dict
        task_config_dict[task_name] = relative_path

    return task_config_dict


