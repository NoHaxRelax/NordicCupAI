// Included inside orchard::Group. Rebuilt after observation ingestion each tick.
bool wi_ready=false;
int wi_nx=0,wi_ny=0;
std::vector<std::vector<int>> wi_cells;
std::vector<int> wi_confirmed;
mutable std::vector<int> wi_all;
mutable std::unordered_map<uint64_t,std::vector<int>> wi_boxes;
