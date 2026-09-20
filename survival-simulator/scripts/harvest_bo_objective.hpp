#pragma once
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace harvest_bo {
struct Result {
    double worst10, worst100, mean, minimum;
    double objective() const { return mean + worst10; }
};
// Only complete, finite 1,000-seed cohorts can produce a ranked BO objective.
inline Result summarize(std::vector<double> scores) {
    if (scores.size() != 1000)
        throw std::invalid_argument("BO objective requires exactly 1000 completed seeds");
    for (double score : scores)
        if (!std::isfinite(score)) throw std::invalid_argument("non-finite game score");
    std::sort(scores.begin(), scores.end());
    auto mean = [&](size_t n) {
        return std::accumulate(scores.begin(), scores.begin() + n, 0.0) / n;
    };
    return {mean(10), mean(100), mean(1000), scores.front()};
}
}
