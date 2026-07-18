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

#include "Data/TraceData.h"
#include "Utility/Errors.h"

#include <stdexcept>

namespace proton {

void TraceData::enterScope(const Scope &scope) { throw NotImplemented(); }

void TraceData::exitScope(const Scope &scope) { throw NotImplemented(); }

size_t TraceData::addOp(size_t scopeId, const std::string &name) {
  throw NotImplemented();
}

void TraceData::addMetric(size_t scopeId, std::shared_ptr<Metric> metric) {
  throw NotImplemented();
}

void TraceData::addMetrics(
    size_t scopeId, const std::map<std::string, MetricValueType> &metrics) {
  throw NotImplemented();
}

void TraceData::clear() { throw NotImplemented(); }

void TraceData::doDump(std::ostream &os, OutputFormat outputFormat) const {
  throw NotImplemented();
}

} // namespace proton
