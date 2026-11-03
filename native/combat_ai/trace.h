#pragma once
#include <cstdint>
#include <initializer_list>
#include <string>
#include <utility>

namespace combat_ai {
// Observational only: no engine queries, callbacks, RNG, or search counters.
// Event and field names are compile-time ASCII identifiers, not external text.
struct DecisionTrace {
  bool enabled = false, truncated = false;
  size_t limit = 64u * 1024u * 1024u;
  int64_t position = 0, candidate = -1;
  std::string data;
  void reset() noexcept {
    data.clear(); truncated = false; position = 0; candidate = -1;
  }
  void event(const char* name,
      std::initializer_list<std::pair<const char*, int64_t>> fields = {}) noexcept {
    if (!enabled || truncated) return;
    try {
      std::string row = "{\"event\":\"" + std::string(name) + "\",\"position\":" +
          std::to_string(position) + ",\"candidate\":" + std::to_string(candidate);
      for (const auto& field : fields)
        row += ",\"" + std::string(field.first) + "\":" + std::to_string(field.second);
      row += "}\n";
      if (data.size() + row.size() > limit) { truncated = true; return; }
      data += row;
    } catch (...) { truncated = true; }
  }
};
} // namespace combat_ai
