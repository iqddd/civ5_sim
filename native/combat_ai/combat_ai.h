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

/* ABI 2. Structs use native alignment, int32 fields and explicit pointers.
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
enum { CAI_ABI = 2, CAI_NO_ID = -1, CAI_MAX_SELECTED = 13, CAI_FIXED_ROLL = 600 };

typedef struct CaiConfig {
  int32_t position_capacity, max_completed, max_branches, choices_per_unit;
  int32_t aggression;
  /* Required explicit opt-in for the CP-source hidden-position queries of v1. */
  int32_t privileged_cp_queries;
} CaiConfig;

/* Encoding matches the simulator: terrain/elevation/feature enums, six river bits. */
typedef struct CaiTile { int32_t terrain, elevation, feature, rivers; } CaiTile;
typedef struct CaiUnit {
  int32_t id, type, side, tile, hp, moves, attacked, acted, fortified, waiting;
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
} CaiSearchRequest;
typedef struct CaiAssignment {
  int32_t kind, unit, from_tile, to_tile, remaining_moves, movement_role;
  int32_t score, damage, self_damage, killed_unit;
} CaiAssignment;
typedef struct CaiSearchStats {
  int32_t checked, generated, completed, pruned, peak_positions;
} CaiSearchStats;
typedef struct CaiSearchResult {
  int32_t reason, assignment_count;
  CaiSearchStats stats;
} CaiSearchResult;
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
   without a partial copy. CP-03 still returns an empty plan. */
CAI_API int32_t cai_search(CaiContext*, const CaiSearchRequest*,
                          CaiAssignment* assignments, int32_t capacity,
                          CaiSearchResult* result);

#ifdef __cplusplus
}
#endif
