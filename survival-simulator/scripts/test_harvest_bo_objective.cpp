#include "harvest_bo_objective.hpp"
#include <cassert>
#include <limits>
int main() {
    std::vector<double> scores(1000);
    std::iota(scores.begin(), scores.end(), 0.0);
    std::reverse(scores.begin(), scores.end());
    auto r = harvest_bo::summarize(scores);
    assert(r.worst10 == 4.5 && r.worst100 == 49.5);
    assert(r.mean == 499.5 && r.minimum == 0.0);
    assert(r.objective() == 504.0);
    scores.back() = -1000;
    assert(harvest_bo::summarize(scores).worst10 == -95.5);
    assert(harvest_bo::summarize(scores).objective() == 403.0);
    scores.back() = std::numeric_limits<double>::quiet_NaN();
    bool rejected = false;
    try { harvest_bo::summarize(scores); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
    scores.pop_back(); rejected = false;
    try { harvest_bo::summarize(scores); } catch (const std::invalid_argument&) { rejected = true; }
    assert(rejected);
}
