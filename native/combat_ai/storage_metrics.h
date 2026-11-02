#pragma once
// Separate diagnostic build only. The timed production DLL has no counters.
namespace combat_ai {
enum StorageMetric {
  ReachHit,
  ReachMiss,
  RangeHit,
  RangeMiss,
  BoardRebuild,
  BoardUpdate,
  BranchAllocation,
  BranchReuse,
  PositionCopy,
  StorageMetricCount
};
#ifdef CAI_STORAGE_METRICS
inline thread_local long long storage_metrics[StorageMetricCount]{};
#define CAI_STORAGE_COUNT(metric) (++::combat_ai::storage_metrics[::combat_ai::metric])
#else
#define CAI_STORAGE_COUNT(metric) ((void)0)
#endif
}  // namespace combat_ai
