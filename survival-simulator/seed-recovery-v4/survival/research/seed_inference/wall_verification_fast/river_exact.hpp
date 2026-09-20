// Copied unchanged from fastsim/_engine.cpp: edge_point and river_path.
    void edge_point(int edge, int64_t& x, int64_t& y) {
        if (edge == 0) { x = rng.randint(0, W - 1); y = 0; }
        else if (edge == 1) { x = rng.randint(0, W - 1); y = H - 1; }
        else if (edge == 2) { x = 0; y = rng.randint(0, H - 1); }
        else { x = W - 1; y = rng.randint(0, H - 1); }
    }
    void river_path(int64_t sx, int64_t sy, int64_t ex, int64_t ey, std::vector<std::pair<int64_t, int64_t>>& path) {
        const double max_turn = 10.0 * (PI / 180.0);
        double direction = np_atan2((double)(ey - sy), (double)(ex - sx));
        int64_t cx = sx, cy = sy;
        int counter = 0;
        while (np_hypot((double)(cx - ex), (double)(cy - ey)) > 5 && counter < 10000) {
            path.push_back({cx, cy});
            direction += rng.uniform(-max_turn, max_turn);
            if (rng.random() < 0.9) {
                double target = np_atan2((double)(ey - cy), (double)(ex - cx));
                direction += 0.3 * (target - direction + PI) / (2 * PI) - PI;
            }
            int64_t step = rng.randint(3, 7);
            int64_t nx = (int64_t)((double)cx + (double)step * np_cos(direction));
            int64_t ny = (int64_t)((double)cy + (double)step * np_sin(direction));
            nx = std::max<int64_t>(0, std::min<int64_t>(W - 1, nx));
            ny = std::max<int64_t>(0, std::min<int64_t>(H - 1, ny));
            if (nx == cx && ny == cy) {
                int64_t dx = ex > cx ? 1 : (ex < cx ? -1 : 0);
                int64_t dy = ey > cy ? 1 : (ey < cy ? -1 : 0);
                nx = std::max<int64_t>(0, std::min<int64_t>(W - 1, cx + dx));
                ny = std::max<int64_t>(0, std::min<int64_t>(H - 1, cy + dy));
            }
            cx = nx; cy = ny;
            counter++;
        }
    }
