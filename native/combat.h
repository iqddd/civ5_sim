#pragma once
#include <cstdint>

#ifdef _WIN32
#define SIM_API extern "C" __declspec(dllexport)
#else
#define SIM_API extern "C" __attribute__((visibility("default")))
#endif

// Version 2 C ABI. Every integer is int32 except the explicit RNG state.
struct Tile {
  int terrain, elevation, feature, rivers;
};
struct Unit {
  int type, side, tile, hp, moves, attacked, acted, fortified, waiting;
  // CP history: defender damage only; rotate when this side starts its turn.
  int damage_taken_this_turn = 0;
  int damage_taken_last_turn = 0;
};
struct Action {
  int kind, unit, target;
};  // move, melee, ranged, fortify, skip, end_turn, swap (target = tile)
struct Result {
  int code, moved, damage, retaliation, attacker_strength, defender_strength;
};
struct Info {
  int side, turn, fog, random, alive0, alive1;
  uint64_t rng;
};

SIM_API int sim_abi();
SIM_API void* sim_new(int w, int h, int parity, int wrap, const Tile*,
                      const Unit*, int count, int fog, int random,
                      uint64_t seed);
SIM_API void sim_free(void*);
SIM_API void* sim_clone(void*);
SIM_API int sim_reset(void*, const Unit*, int count, const Info*);
SIM_API int sim_units(void*, int observer, Unit*);  // observer=-1 is privileged
SIM_API int sim_visibility(void*, int side, int*);
// Additive ABI 2 export: restore both public masks, preserving current sources.
SIM_API int sim_restore_visibility(void*, const int* masks, int count);
SIM_API int sim_visibility_history(void*, int side, int*);
SIM_API int sim_info(void*, Info*);
SIM_API int sim_actions(void*, Action*, int capacity);
SIM_API Result sim_step(void*, Action);
SIM_API Result sim_preview(void*, Action);
SIM_API void sim_batch(void** states, const Action*, Result*, int count);
SIM_API int sim_damage(int attack, int defense,
                       int roll);  // roll 0..1199, 600 = fixed
SIM_API const char* sim_rules();
SIM_API uint64_t sim_memory(void*, int include_geometry);
