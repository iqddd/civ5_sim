#pragma once
#include <stdint.h>

#ifdef _WIN32
#define CAI_API __declspec(dllexport)
#else
#define CAI_API __attribute__((visibility("default")))
#endif
#ifdef __cplusplus
extern "C" {
#endif

/* ABI 6. Structs use native alignment, int32 fields and explicit pointers.
   All input arrays are borrowed for one synchronous call; no RNG is accepted.
   A context is owned by its caller and must not be used concurrently. */
enum CaiStatus { CAI_OK, CAI_BAD_ARGUMENT, CAI_INTERNAL_ERROR, CAI_BUFFER_TOO_SMALL };
enum CaiReason {
  CAI_COMPLETED, CAI_NO_PLAN, CAI_BUDGET_EXHAUSTED, CAI_TURN_CHANGED,
  CAI_INVALID_REQUEST, CAI_NOT_IMPLEMENTED
};
enum CaiAggression { CAI_NONE, CAI_LOW, CAI_MEDIUM, CAI_HIGH, CAI_BRAVEHEART };
enum CaiAssignmentKind {
  CAI_INITIAL, CAI_MOVE, CAI_MELEE, CAI_RANGED, CAI_FINISH, CAI_BLOCKED,
  CAI_RESTART, CAI_SWAP, CAI_SWAP_REVERSE, CAI_FINISH_TEMP
};
enum CaiMovementRole { CAI_ROLE_NONE, CAI_FIRST_LINE, CAI_SECOND_LINE, CAI_THIRD_LINE };
enum CaiSupportMode { CAI_HEAL_BEFORE, CAI_HEAL_AFTER, CAI_APPROACH, CAI_RECOVERY };
enum CaiRecoveryKind { CAI_RECOVERY_NONE, CAI_RECOVERY_ATTACK, CAI_RECOVERY_RETREAT, CAI_RECOVERY_WAIT };
enum { CAI_ABI = 6, CAI_NO_ID = -1, CAI_MAX_SELECTED = 13, CAI_FIXED_ROLL = 600 };

typedef struct CaiConfig {
  int32_t position_capacity, max_completed, max_branches, choices_per_unit;
  int32_t aggression;
  /* Required explicit opt-in for the CP-source hidden-position queries of v1. */
  int32_t privileged_cp_queries;
  int32_t risk_budget_base, risk_budget_per_kill; /* Each 0..13. */
} CaiConfig;

/* Encoding matches the simulator: terrain/elevation/feature enums, six river bits. */
typedef struct CaiTile { int32_t terrain, elevation, feature, rivers; } CaiTile;
typedef struct CaiUnit {
  int32_t id, type, side, tile, hp, moves, attacked, acted, fortified, waiting;
  int32_t damage_taken_last_turn; /* CP defender-damage history, nonnegative. */
} CaiUnit;
typedef struct CaiBattleState {
  int32_t width, height, row_parity, wrap_x;
  const CaiTile* tiles;
  int32_t tile_count;
  const CaiUnit* units;  /* Complete stable roster, including hp=0 slots. */
  int32_t unit_count;
  const int32_t* visibility; /* Boolean mask for the requesting side. */
  int32_t visibility_count;
  int32_t active_side, turn, fog;
} CaiBattleState;
typedef struct CaiContact { int32_t id, last_seen_own_turn; } CaiContact;
typedef struct CaiContactMemory {
  int32_t turn; /* Last real observation update, <= state.turn. */
  int32_t side; /* Observer, or -1 only for the default empty memory at turn 0. */
  const CaiContact* known_enemies;
  int32_t known_enemy_count;
} CaiContactMemory;
typedef struct CaiSearchRequest {
  const CaiBattleState* state;
  int32_t side, expected_turn, target;
  const int32_t* selected_units;
  int32_t selected_count;
  CaiContactMemory contacts;
  int32_t aggression_override; /* 0 uses context config; otherwise CaiAggression + 1. */
} CaiSearchRequest;
typedef struct CaiAssignment {
  int32_t kind, unit, from_tile, to_tile, remaining_moves, movement_role;
  int32_t score, damage, self_damage, killed_unit;
  int32_t command_target; /* Original command destination; -1 for bookkeeping. */
} CaiAssignment;
typedef struct CaiSearchStats {
  int32_t checked, generated, completed, pruned, peak_positions;
} CaiSearchStats;
typedef struct CaiSearchResult {
  int32_t reason, assignment_count;
  CaiSearchStats stats;
} CaiSearchResult;
/* Additive ABI6 diagnostics; result/layout and existing calls are unchanged.
   Counts are evidence encountered by this bounded search, not a proof of why
   every possible plan fails. IDs are sorted, unique, and selected by the caller. */
typedef struct CaiSearchDiagnostics {
  int32_t budget_exhausted, risk_rejections, improvement_rejections;
  int32_t bad_unit_count, bad_units[13], recovery_kind;
} CaiSearchDiagnostics;
typedef struct CaiContext CaiContext;

CAI_API int32_t cai_abi(void);
CAI_API const char* cai_version(void); /* Library-owned static UTF-8 string. */
CAI_API CaiConfig cai_default_config(void);
CAI_API CaiContext* cai_new(const CaiConfig* config); /* NULL on invalid config/allocation failure. */
CAI_API void cai_free(CaiContext* context); /* NULL is allowed. */
/* Invalid search contents return CAI_OK with reason=CAI_INVALID_REQUEST.
   Invalid API pointers/capacity return CAI_BAD_ARGUMENT. result is overwritten
   on every call when non-NULL. Future plans are copied to caller-owned storage;
   insufficient capacity returns CAI_BUFFER_TOO_SMALL and the required count,
   without a partial copy. Budgets apply independently to each call. */
CAI_API int32_t cai_search(CaiContext*, const CaiSearchRequest*,
                          CaiAssignment* assignments, int32_t capacity,
                          CaiSearchResult* result);
CAI_API int32_t cai_support(CaiContext*, const CaiSearchRequest*, int32_t mode,
                           CaiAssignment* assignments, int32_t capacity,
                           CaiSearchResult* result);
/* Most recent call, overwritten even after an invalid request. */
CAI_API int32_t cai_search_diagnostics(const CaiContext*, CaiSearchDiagnostics*);

#ifdef __cplusplus
}
#endif
