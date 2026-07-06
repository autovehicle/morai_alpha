from pathlib import Path
import sys
import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
RUNNER_ROOT = PROJECT_ROOT / "third_party" / "morai_scenario_runner"
RUNNER_PACKAGE_DIR = RUNNER_ROOT / "scenario_runner"
RUNNER_CONFIG_DIR = RUNNER_PACKAGE_DIR / "config"


def ensure_scenario_runner_on_path():
    if not RUNNER_PACKAGE_DIR.exists():
        raise RuntimeError(
            "External scenario runner is missing. "
            "Run: git submodule update --init --recursive"
        )

    for path in (RUNNER_PACKAGE_DIR, RUNNER_ROOT):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)


def deep_update(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_runner_paths(cfg: dict) -> dict:
    paths = cfg.setdefault("paths", {})
    for key, value in list(paths.items()):
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                paths[key] = str(RUNNER_ROOT / path)
    return cfg


def load_global_cfg() -> dict:
    cfg = load_yaml(RUNNER_CONFIG_DIR / "global.yaml")
    local_path = RUNNER_CONFIG_DIR / "local.yaml"
    if local_path.exists():
        deep_update(cfg, load_yaml(local_path))
    return resolve_runner_paths(cfg)


def apply_collection_morai_overrides(global_cfg: dict, collection_cfg: dict) -> dict:
    morai_cfg = collection_cfg.get("morai", {})
    grpc_cfg = global_cfg.setdefault("grpc", {})
    runner_morai_cfg = global_cfg.setdefault("morai", {})

    if morai_cfg.get("host"):
        grpc_cfg["host"] = morai_cfg["host"]
    if morai_cfg.get("grpc_port") is not None:
        grpc_cfg["port"] = int(morai_cfg["grpc_port"])
    if morai_cfg.get("ego_vehicle"):
        runner_morai_cfg["ego_vehicle_model"] = morai_cfg["ego_vehicle"]

    return global_cfg


def load_zone_cfg(zone: str) -> dict:
    cfg = load_yaml(RUNNER_CONFIG_DIR / f"{zone}.yaml")
    for scenario_cfg in cfg.get("scenarios", {}).values():
        resolve_scenario_paths(scenario_cfg)
    return cfg


def resolve_scenario_paths(scenario_cfg: dict) -> dict:
    for key in ("zone_links_path", "gt_bev_repo_path", "route_debug_dir"):
        value = scenario_cfg.get(key)
        if isinstance(value, str):
            path = Path(value)
            if not path.is_absolute():
                scenario_cfg[key] = str(RUNNER_ROOT / path)
    return scenario_cfg


def create_scenario(zone: str, scenario: str, grpc_client, map_loader, global_cfg, scenario_cfg):
    ensure_scenario_runner_on_path()

    if zone != "urban":
        raise ValueError(f"Unknown scenario: {zone}/{scenario}")

    from scenario_runner.zones.urban_scenarios import (
        UrbanBasicDriveScenario,
        UrbanPedestrianYieldScenario,
        UrbanSuddenBrakeExpertScenario,
        UrbanTrafficJamScenario,
    )

    scenario_classes = {
        "basic_drive": UrbanBasicDriveScenario,
        "sudden_brake": UrbanSuddenBrakeExpertScenario,
        "traffic_jam": UrbanTrafficJamScenario,
        "bottleneck": UrbanTrafficJamScenario,
        "pedestrian_yield": UrbanPedestrianYieldScenario,
        "yield_pedestrian": UrbanPedestrianYieldScenario,
    }

    try:
        scenario_cls = scenario_classes[scenario]
    except KeyError as exc:
        raise ValueError(f"Unknown scenario: {zone}/{scenario}") from exc

    return scenario_cls(
        grpc_client=grpc_client,
        map_loader=map_loader,
        global_cfg=global_cfg,
        scenario_cfg=scenario_cfg,
    )
