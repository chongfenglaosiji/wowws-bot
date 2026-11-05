import ast
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main_source():
    return (ROOT / "main.py").read_text(encoding="utf-8")


class PortConfigurationLifecycleTests(unittest.TestCase):
    def test_port_configuration_is_committed_before_hud_wait(self):
        source = main_source()
        tree = ast.parse(source)
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "battle_queued"
                for target in node.targets
            )
        ]
        self.assertTrue(assignments)
        self.assertIn("configure_port=configured_this_attempt", source)
        # The HUD wait must consume the same queue result that triggered port
        # configuration; no second, stale port-selection state may exist.
        self.assertLess(
            source.index("battle_queued = prepare_battle("),
            source.index("prepared = battle_queued and wait_for_battle"),
        )
        self.assertNotIn("port_configured", source)

    def test_ship_lock_and_pause_recheck_gate_port_configuration(self):
        source = main_source()
        self.assertIn("configured_this_attempt = not ship_locked or bool(", source)
        self.assertIn('getattr(bot, "_port_recheck_required", False)', source)


if __name__ == "__main__":
    unittest.main()
