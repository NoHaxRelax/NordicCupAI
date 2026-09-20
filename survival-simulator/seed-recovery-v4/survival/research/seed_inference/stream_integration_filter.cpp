// Bounded fixture only. Production uses the unmodified full-domain terrain_filter.
#include <unistd.h>
#include <filesystem>
#include <vector>
#include <string>
int main(int argc,char**argv){if(argc<6)return 2;std::vector<std::string>a;
 a.push_back((std::filesystem::path(argv[0]).parent_path()/"terrain_filter").string());
 for(int i=1;i<argc;i++)a.push_back(argv[i]);a[2]="1853968307";a[3]="1048576";
 std::vector<char*>v;for(auto&s:a)v.push_back(s.data());v.push_back(nullptr);execv(v[0],v.data());return 127;}
