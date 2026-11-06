#pragma once
// Opt-in inclusive CPU timers. Production builds compile these scopes out.
#ifdef CAI_PROFILE
#include <chrono>
#include <cstdint>
namespace combat_ai {
enum ProfileMetric {
  GeometryBuild, RootImport, StateExport, TacticalViewBuild, BoardRefresh,
  ValidateRequest, SearchCall, SupportCall, CandidatesCall, ProfileCount
};
inline thread_local uint64_t profile_ns[ProfileCount]{};
inline thread_local uint64_t profile_calls[ProfileCount]{};
inline constexpr const char* profile_names[] = {
  "geometry_build", "root_import", "state_export", "tactical_view", "board_refresh",
  "validate_request", "search_call", "support_call", "candidates_call"
};
class ProfileScope {
  ProfileMetric metric;
  std::chrono::steady_clock::time_point start;
 public:
  explicit ProfileScope(ProfileMetric value) : metric(value), start(std::chrono::steady_clock::now()) {}
  ~ProfileScope() {
    profile_ns[metric] += uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count());
    ++profile_calls[metric];
  }
};
}
#define CAI_PROFILE_SCOPE(metric) ::combat_ai::ProfileScope profile_scope_##metric(::combat_ai::metric)
#else
#define CAI_PROFILE_SCOPE(metric) ((void)0)
#endif
