from __future__ import annotations

import math
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metaumbra.scoring import GenomePresenceScorer, _effective_unique_count


class UniqueEmpiricalBackgroundTests(unittest.TestCase):
    def _run_empirical_auto_fit(self, rows):
        out = pd.DataFrame(rows).copy()
        out["p_shared_knock"] = pd.to_numeric(out.get("p_shared_knock", 1.0), errors="coerce").fillna(1.0)
        out["p_presence"] = 1.0
        out["q_presence"] = 1.0
        target_mask = out["_genomes_with_any_match"].astype(bool)
        target_df = out.loc[target_mask].copy()

        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unique_pvalue_mode = "empirical-background"
        scorer.single_peptide_error_rate_upper_bound = 0.05
        scorer._fit_unique_empirical_background_auto(
            out=out,
            target_df=target_df,
            target_mask=target_mask,
        )
        return scorer, out

    def test_empirical_background_downweights_deep_weak_unique_counts(self) -> None:
        rows = []
        for i in range(120):
            u = 3 + (i % 4)
            rows.append(
                {
                    "genome_id": f"bg_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                }
            )
        for gid, u in (("strong_a", 22), ("strong_b", 25)):
            rows.append(
                {
                    "genome_id": gid,
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                }
            )

        scorer = GenomePresenceScorer(num_workers=1)
        scorer.single_peptide_error_rate_upper_bound = 0.05
        scorer._prepare_unique_empirical_background(pd.DataFrame(rows), n_bins=4, min_bin_size=50)

        weak_pvalues = [
            scorer.unique_empirical_pvalue_by_genome[f"bg_{i:03d}"]
            for i in range(120)
        ]
        self.assertGreaterEqual(min(weak_pvalues), 1.0)
        self.assertLess(scorer.unique_empirical_pvalue_by_genome["strong_a"], 1e-6)
        self.assertLess(scorer.unique_empirical_pvalue_by_genome["strong_b"], 1e-6)
        self.assertGreater(scorer.unique_empirical_tail_by_genome["strong_a"], 0.001)
        self.assertEqual(scorer.run_stats["unique_empirical_background_excluded_genomes"], 13)
        self.assertAlmostEqual(
            scorer.run_stats["unique_empirical_background_excluded_fraction"],
            13 / 122,
        )
        self.assertEqual(
            scorer.run_stats["unique_empirical_background_opportunity_source"],
            "total_peptide_count",
        )

    def test_empirical_background_auto_keeps_small_dataset_near_initial_exclusion(self) -> None:
        rows = []
        for i in range(80):
            u = 3 + (i % 4)
            rows.append(
                {
                    "genome_id": f"bg_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1.0,
                }
            )
        for i in range(4):
            u = 20 + i
            rows.append(
                {
                    "genome_id": f"hit_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1.0,
                }
            )

        scorer, out = self._run_empirical_auto_fit(rows)

        self.assertAlmostEqual(
            scorer.run_stats["unique_empirical_background_final_exclude_fraction"],
            0.10,
        )
        self.assertEqual(scorer.run_stats["unique_empirical_background_exclude_mode"], "auto")
        self.assertLessEqual(scorer.run_stats["unique_empirical_background_iterations"], 3)
        self.assertTrue((out.loc[out["genome_id"].str.startswith("bg_"), "p_unique"] == 1.0).all())

    def test_empirical_background_auto_increases_exclusion_for_cohort_like_candidates(self) -> None:
        rows = []
        for i in range(140):
            u = 3 + (i % 4)
            rows.append(
                {
                    "genome_id": f"bg_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1.0,
                }
            )
        for i in range(60):
            u = 50 + (i % 3)
            rows.append(
                {
                    "genome_id": f"hit_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1e-8,
                }
            )

        scorer, out = self._run_empirical_auto_fit(rows)
        final_fraction = scorer.run_stats["unique_empirical_background_final_exclude_fraction"]

        self.assertGreater(final_fraction, 0.10)
        self.assertLessEqual(final_fraction, scorer.unique_empirical_background_max_exclude_fraction)
        self.assertAlmostEqual(final_fraction, 0.30)
        self.assertLess(out.loc[out["genome_id"] == "hit_000", "p_unique"].iloc[0], 1e-6)
        self.assertTrue((out.loc[out["genome_id"].str.startswith("bg_"), "q_presence"] > 0.20).all())

    def test_empirical_background_auto_respects_max_exclude_fraction(self) -> None:
        rows = []
        for i in range(100):
            u = 3 + (i % 4)
            rows.append(
                {
                    "genome_id": f"bg_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1.0,
                }
            )
        for i in range(100):
            u = 50 + (i % 3)
            rows.append(
                {
                    "genome_id": f"hit_{i:03d}",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": u,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": float(u),
                    "weighted_evidence": float(u),
                    "p_shared_knock": 1e-8,
                }
            )

        out = pd.DataFrame(rows).copy()
        out["p_presence"] = 1.0
        out["q_presence"] = 1.0
        target_mask = out["_genomes_with_any_match"].astype(bool)
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unique_pvalue_mode = "empirical-background"
        scorer.single_peptide_error_rate_upper_bound = 0.05
        scorer.unique_empirical_background_max_exclude_fraction = 0.20
        scorer._fit_unique_empirical_background_auto(
            out=out,
            target_df=out.loc[target_mask].copy(),
            target_mask=target_mask,
        )

        self.assertEqual(scorer.run_stats["unique_empirical_background_final_exclude_fraction"], 0.20)

    def test_empirical_unique_stats_report_background_diagnostics(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "genome_id": "bg",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": 4,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": 4.0,
                    "weighted_evidence": 4.0,
                },
                {
                    "genome_id": "hit",
                    "_genomes_with_any_match": True,
                    "num_peptides_unique": 20,
                    "total_peptide_count": 1000,
                    "unique_weighted_evidence": 20.0,
                    "weighted_evidence": 20.0,
                },
            ]
        )
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unique_pvalue_mode = "empirical-background"
        scorer.single_peptide_error_rate_upper_bound = 0.05
        scorer._prepare_unique_empirical_background(df, top_exclude_fraction=0.0, n_bins=1)

        stats = scorer._unique_pvalue_stats_for_genome("hit", 20)

        self.assertEqual(stats["unique_depth_null_model"], "empirical-background")
        self.assertEqual(stats["unique_pvalue_count_model"], "background-excess")
        self.assertEqual(stats["unique_empirical_background_bin"], "bin_0")
        self.assertEqual(stats["unique_empirical_background_size"], 2)
        self.assertGreater(stats["unique_empirical_background_threshold"], 0)
        self.assertGreater(stats["unique_empirical_excess_count"], 0)
        self.assertEqual(stats["unique_effective_count"], stats["unique_empirical_excess_count"])
        self.assertEqual(stats["p_unique"], scorer.unique_empirical_pvalue_by_genome["hit"])
        self.assertEqual(stats["p_unique_empirical_tail"], scorer.unique_empirical_tail_by_genome["hit"])

    def test_alpha_upper_bound_stats_are_unchanged_when_configured(self) -> None:
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unique_pvalue_mode = "alpha-upper-bound"
        scorer.unique_peptide_error_source = "global-alpha"
        scorer.unique_count_power = 0.7
        scorer.single_peptide_error_rate_upper_bound = 0.05
        scorer.genome_matched_peptides = {"g": {"p1", "p2", "p3", "p4"}}
        scorer.peptide_degeneracy = {"p1": 1, "p2": 1, "p3": 1, "p4": 1}

        stats = scorer._unique_pvalue_stats_for_genome("g", 4)
        expected_eff = _effective_unique_count(4, 0.7)

        self.assertTrue(math.isclose(stats["unique_effective_count"], expected_eff))
        self.assertTrue(math.isclose(stats["p_unique"], 0.05 ** expected_eff))
        self.assertEqual(stats["unique_pvalue_count_model"], "power:0.7")

    def test_hypergeometric_opportunity_stats_are_unchanged_when_configured(self) -> None:
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unique_pvalue_mode = "hypergeometric-opportunity"
        scorer.observed_unique_peptide_pool_size = 10
        scorer.genome_theoretical_unique_peptides = {"g": 20}
        scorer.total_theoretical_unique_peptides_all_genomes = 100

        def fake_tail(observed: int, universe_size: int, success_states: int, draws: int) -> float:
            self.assertEqual((observed, universe_size, success_states, draws), (3, 100, 20, 10))
            return 0.123

        scorer._hypergeom_tail_pvalue = fake_tail  # type: ignore[method-assign]
        stats = scorer._unique_pvalue_stats_for_genome("g", 3)

        self.assertEqual(stats["p_unique"], 0.123)
        self.assertEqual(stats["unique_depth_null_model"], "hypergeometric")
        self.assertTrue(math.isclose(stats["unique_expected_null"], 2.0))
        self.assertTrue(math.isclose(stats["unique_depth_fold"], 1.5))

    def test_empirical_background_rejects_unit_aware_for_now(self) -> None:
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.unit_aware_enabled = True

        with self.assertRaisesRegex(ValueError, "not currently supported with unit_aware=True"):
            scorer.analyze_genomes(
                genome_digest_dirs=[],
                output_tsv_path="unused.tsv",
                all_matched_peptides=[],
                unique_pvalue_mode="empirical-background",
                unit_aware=True,
            )

    def test_empirical_background_does_not_build_theoretical_opportunity_by_default(self) -> None:
        scorer = GenomePresenceScorer(num_workers=1)
        scorer.peptide_score = {"bg1": 1.0, "bg2": 1.0, "hit1": 1.0, "hit2": 1.0, "hit3": 1.0}

        def fail_if_called(*args, **kwargs):
            raise AssertionError("empirical-background should not build theoretical opportunity by default")

        scorer._load_or_build_theoretical_opportunity = fail_if_called  # type: ignore[method-assign]
        all_matched = [
            ("bg_a", {"bg1"}, 100),
            ("bg_b", {"bg2"}, 100),
            ("hit", {"hit1", "hit2", "hit3"}, 100),
        ]

        with tempfile.TemporaryDirectory() as tmp:
            result = scorer.analyze_genomes(
                genome_digest_dirs=[],
                output_tsv_path=str(Path(tmp) / "presence.tsv"),
                all_matched_peptides=all_matched,
                save_matched_peptides_cache=False,
                use_cache_if_exists=False,
                compute_coverage=False,
                export_temp=False,
                unique_pvalue_mode="empirical-background",
                return_full_table=True,
            )

        self.assertFalse(result.empty)
        self.assertEqual(scorer.genome_theoretical_unique_peptides, {})
        self.assertEqual(scorer.total_theoretical_unique_peptides_all_genomes, 0)
        self.assertEqual(scorer.theoretical_peptide_universe_size, 0)
        self.assertIsNone(scorer.run_stats["theoretical_opportunity_cache_path"])
        self.assertFalse(scorer.run_stats["theoretical_opportunity_cache_rebuilt"])
        self.assertEqual(
            scorer.run_stats["unique_empirical_background_opportunity_source"],
            "total_peptide_count",
        )


if __name__ == "__main__":
    unittest.main()
