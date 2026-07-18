// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "Proton.h"

#include <map>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

using namespace proton;

static void initProton(pybind11::module &&m) {
  using ret = pybind11::return_value_policy;
  using namespace pybind11::literals;

  m.def("start",
        [](const std::string &path, const std::string &contextSourceName,
           const std::string &dataName, const std::string &profilerName,
           const std::string &profilerPath) {
          auto sessionId = SessionManager::instance().addSession(
              path, profilerName, profilerPath, contextSourceName, dataName);
          SessionManager::instance().activateSession(sessionId);
          return sessionId;
        });

  m.def("activate", [](size_t sessionId) {
    SessionManager::instance().activateSession(sessionId);
  });

  m.def("activate_all",
        []() { SessionManager::instance().activateAllSessions(); });

  m.def("deactivate", [](size_t sessionId) {
    SessionManager::instance().deactivateSession(sessionId);
  });

  m.def("deactivate_all",
        []() { SessionManager::instance().deactivateAllSessions(); });

  m.def("finalize", [](size_t sessionId, const std::string &outputFormat) {
    auto outputFormatEnum = parseOutputFormat(outputFormat);
    SessionManager::instance().finalizeSession(sessionId, outputFormatEnum);
  });

  m.def("finalize_all", [](const std::string &outputFormat) {
    auto outputFormatEnum = parseOutputFormat(outputFormat);
    SessionManager::instance().finalizeAllSessions(outputFormatEnum);
  });

  m.def("record_scope", []() { return Scope::getNewScopeId(); });

  m.def("enter_scope", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().enterScope(Scope(scopeId, name));
  });

  m.def("exit_scope", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().exitScope(Scope(scopeId, name));
  });

  m.def("enter_op", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().enterOp(Scope(scopeId, name));
  });

  m.def("exit_op", [](size_t scopeId, const std::string &name) {
    SessionManager::instance().exitOp(Scope(scopeId, name));
  });

  m.def("enter_state", [](const std::string &state) {
    SessionManager::instance().setState(state);
  });

  m.def("exit_state",
        []() { SessionManager::instance().setState(std::nullopt); });

  m.def("add_metrics",
        [](size_t scopeId,
           const std::map<std::string, MetricValueType> &metrics) {
          SessionManager::instance().addMetrics(scopeId, metrics);
        });

  m.def("get_context_depth", [](size_t sessionId) {
    return SessionManager::instance().getContextDepth(sessionId);
  });

  pybind11::bind_map<std::map<std::string, MetricValueType>>(m, "MetricMap");
}

PYBIND11_MODULE(libproton, m) {
  m.doc() = "Python bindings to the Proton API";
  initProton(std::move(m.def_submodule("proton")));
}
