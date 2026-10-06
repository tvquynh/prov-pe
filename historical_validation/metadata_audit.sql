-- Read-only audit of one terminal response per SHA-256 in the completed snapshot.
-- Engine-category totals come from last_analysis_stats stored as stats_json.
-- They are not counts of normalized individual last_analysis_results records.
-- QUERY terminal_status_counts
SELECT terminal_status, COUNT(*) AS samples
FROM snapshot_samples
WHERE snapshot_id = 'COLLECTED_PE_779619_20260923'
GROUP BY terminal_status
ORDER BY terminal_status;

-- QUERY engine_roster_and_stats_summary
SELECT COUNT(*) AS terminal_ok_reports,
       MIN(engine_roster_size) AS min_engine_roster_size,
       MAX(engine_roster_size) AS max_engine_roster_size,
       MIN(stats_total) AS min_stats_total,
       MAX(stats_total) AS max_stats_total,
       SUM(engine_roster_size <> stats_total) AS reports_with_roster_stats_difference,
       SUM(engine_roster_size = 0) AS reports_with_zero_roster,
       SUM(stats_total = 0) AS reports_with_zero_stats_total
FROM snapshot_samples
WHERE snapshot_id = 'COLLECTED_PE_779619_20260923'
  AND terminal_status = 'ok';

-- QUERY missing_stats_json
SELECT COUNT(*) AS terminal_ok_reports,
       SUM(stats_json IS NULL) AS reports_with_missing_stats_json
FROM snapshot_samples
WHERE snapshot_id = 'COLLECTED_PE_779619_20260923'
  AND terminal_status = 'ok';

-- QUERY stats_category_totals
SELECT category.key AS category,
       SUM(category.value) AS sum_of_reported_category_counts,
       SUM(CASE WHEN category.value > 0 THEN 1 ELSE 0 END) AS samples_with_category_positive
FROM snapshot_samples AS sample, json_each(sample.stats_json) AS category
WHERE sample.snapshot_id = 'COLLECTED_PE_779619_20260923'
  AND sample.terminal_status = 'ok'
GROUP BY category.key
ORDER BY category.key;
