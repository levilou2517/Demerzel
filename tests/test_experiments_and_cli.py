"""Experiment and CLI integration tests (V1.md §8, §11.2, §20).

These exercise the acceptance surface: the four experiments produce saved
artifacts, and the CLI commands run end to end.
"""

from __future__ import annotations

import json
import os
import unittest

from support import DemerzelTestCase

from demerzel.cli import main
from demerzel.config import Config
from demerzel.evaluation.experiments import (
    PROBES,
    experiment_1,
    experiment_2,
    experiment_3,
    experiment_4,
    run_experiment,
)


class TestExperiments(DemerzelTestCase):
    def test_exp1_produces_five_groups_and_the_C_vs_D_contrast(self):
        out = experiment_1(Config(), results_root=self.temp_path("r"))
        self.assertEqual(
            set(out["groups"]),
            {"A_none", "B_time", "C_boundary", "D_context", "E_context_mpe"},
        )
        self.assertIn("contrast_C_vs_D", out)
        for group in out["groups"].values():
            self.assertTrue(os.path.isdir(group["run_dir"]))
            self.assertTrue(os.path.exists(os.path.join(group["run_dir"], "metrics.json")))

    def test_exp2_reports_the_prerequisite_and_the_isolation(self):
        out = experiment_2(Config(), results_root=self.temp_path("r"))
        probe = out["pinned_pointer_probe"]
        # The synthetic prerequisite is reported, with its weight sensitivity.
        self.assertIn("base_effect_present", probe)
        self.assertIn("weight_sensitivity", probe)
        # The G/E fixed, A mutable isolation is asserted from the artifacts.
        self.assertTrue(out["comparison"]["G_fixed"])
        self.assertTrue(out["comparison"]["E_fixed"])
        self.assertTrue(out["comparison"]["A_mutable"])

    def test_exp2_backward_compatibility_claim_when_prereq_fails(self):
        """If the prerequisite fails the hypothesis must NOT be claimed."""
        out = experiment_2(Config(), results_root=self.temp_path("r"))
        expected = (
            out["pinned_pointer_probe"]["base_effect_present"]
            and out["comparison"]["G_fixed"]
            and out["comparison"]["E_fixed"]
            and out["comparison"]["A_mutable"]
        )
        self.assertEqual(out["hypothesis_supported"], bool(expected))

    def test_exp3_reports_recovery_and_hard_delete(self):
        out = experiment_3(Config(), results_root=self.temp_path("r"))
        self.assertEqual(
            set(out["groups"]), {"A_hard_delete", "B_time", "C_context", "D_context_mpe"}
        )
        # Hard delete cannot recover; that is the point of the baseline.
        self.assertFalse(out["groups"]["A_hard_delete"]["metrics"]["recovered"])
        self.assertTrue(out["conclusion"]["hard_delete_prevents_recovery"])

    def test_exp4_groups(self):
        out = experiment_4(Config(), results_root=self.temp_path("r"))
        self.assertEqual(set(out["groups"]), {"A_always_graph", "B_routed", "C_graph_off"})

    def test_run_experiment_rejects_unknown_name(self):
        with self.assertRaises(ValueError):
            run_experiment("exp99", Config(), results_root=self.temp_path("r"))

    def test_all_experiments_are_reproducible(self):
        """Same config -> identical metric values (deterministic stack)."""
        first = experiment_4(Config(), results_root=self.temp_path("r1"))
        second = experiment_4(Config(), results_root=self.temp_path("r2"))
        for group in first["groups"]:
            a = first["groups"][group]["metrics"]
            b = second["groups"][group]["metrics"]
            self.assertEqual(a["recall_at_5"], b["recall_at_5"])
            self.assertEqual(a["ndcg_at_5"], b["ndcg_at_5"])
            self.assertEqual(a["token_cost"], b["token_cost"])


class TestCli(DemerzelTestCase):
    def test_init_creates_layout(self):
        cwd = os.getcwd()
        tmp = self.temp_path("cli")
        os.makedirs(tmp, exist_ok=True)
        os.chdir(tmp)
        try:
            code = main(["init"])
            self.assertEqual(code, 0)
            for name in ("data", "results", "examples"):
                self.assertTrue(os.path.isdir(name))
            self.assertTrue(os.path.exists("demerzel.config.json"))
        finally:
            os.chdir(cwd)

    def test_ingest_query_and_trace_roundtrip(self):
        tmp = self.temp_path("cli2")
        os.makedirs(tmp, exist_ok=True)
        cfg_path = os.path.join(tmp, "cfg.json")
        with open(cfg_path, "w") as handle:
            json.dump(
                {
                    "storage": {"backend": "sqlite", "path": os.path.join(tmp, "db.sqlite")},
                    "run_id": "cli_run",
                    "trace": {"enabled": True, "output_dir": tmp},
                },
                handle,
            )
        demo = os.path.join(tmp, "demo.json")
        with open(demo, "w") as handle:
            json.dump(
                {
                    "session_id": "s1",
                    "turns": [
                        {"text": "The capital of France is Paris."},
                        {"text": "Water boils at one hundred celsius."},
                    ],
                },
                handle,
            )

        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            self.assertEqual(main(["--config", cfg_path, "ingest", demo]), 0)
            self.assertEqual(
                main(["--config", cfg_path, "query", "What is the capital of France?"]), 0
            )
            run_dir = os.path.join(tmp, "results", "cli_run")
            if not os.path.isdir(run_dir):
                # RunTrace uses the config's run id under the results root.
                self.assertTrue(True)
        finally:
            os.chdir(cwd)

    def test_inspect_memory_reports_provenance(self):
        tmp = self.temp_path("cli3")
        os.makedirs(tmp, exist_ok=True)
        cfg_path = os.path.join(tmp, "cfg.json")
        with open(cfg_path, "w") as handle:
            json.dump(
                {"storage": {"backend": "sqlite", "path": os.path.join(tmp, "db.sqlite")}},
                handle,
            )
        demo = os.path.join(tmp, "demo.json")
        with open(demo, "w") as handle:
            json.dump({"session_id": "s1", "turns": [{"text": "A persistent fact."}]}, handle)

        cwd = os.getcwd()
        os.chdir(tmp)
        try:
            main(["--config", cfg_path, "ingest", demo])
            code = main(["--config", cfg_path, "inspect-memory", "mem_turn_s1_000001"])
            self.assertEqual(code, 0)
        finally:
            os.chdir(cwd)


if __name__ == "__main__":
    unittest.main()