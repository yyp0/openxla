#ifndef XLA_SERVICE_PASS_CONTEXT_H_
#define XLA_SERVICE_PASS_CONTEXT_H_

#include <string>
#include <utility>

#include "pybind11/pybind11.h"
#include "xla/util.h"
#include "xla/types.h"

namespace xla {

// A global context to pass arguments from python to xla passes.
namespace pass_context {

// Read context values from a python dict.
void SetPassContext(pybind11::dict dict);

// Clear context values.
void ClearPassContext();

int64_t GetInt(const std::string& name, int64_t default_value);

bool GetBool(const std::string& name, bool default_value);

double GetDouble(const std::string& name);

std::string GetString(const std::string& name, const std::string& default_value);

std::vector<int64_t> GetIntVector(const std::string& name);

std::vector<double> GetDoubleVector(const std::string& name);

std::vector<std::string> GetStringVector(const std::string& name);

pybind11::object GetPyObject(const std::string& name);

}  // namespace pass_context
}  // namespace xla

#endif  // XLA_SERVICE_PASS_CONTEXT_H_
