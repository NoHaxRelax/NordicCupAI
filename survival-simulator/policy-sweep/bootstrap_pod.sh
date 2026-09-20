#!/bin/bash
# Self-contained pod setup for the seed-aware policy sweep.
# Assumes /root/sap.tar has already been uploaded (git archive of
# origin/codex/seed-aware-policy-2200 survival-simulator/seed-aware-policy).
set -e
cd /root
tar xf sap.tar
cd /root/survival-simulator/seed-aware-policy

# 1. the committed bench points at a path that does not exist on this branch
sed -i 's#../lucas_integration/native/_nengine.cpp#../native/_nengine.cpp#' bench/policy_bench.cpp
# 2. build.py hardcodes python3.12; this image ships 3.11
PYV=$(python3 -c 'import sys;print(f"{sys.version_info.major}.{sys.version_info.minor}")')
sed -i "s/-lpython3.12/-lpython${PYV}/" build.py
# 3. headers that older GCC pulled in transitively
for f in native/fruit_assignment.hpp native/model_world.hpp native/resource_integration.hpp \
         native/entrapment_guide.hpp native/native_corner.hpp native/_npolicy.hpp; do
  [ -f "$f" ] || continue
  grep -q '#include <map>'   "$f" || sed -i '1i #include <map>'   "$f"
  grep -q '#include <set>'   "$f" || sed -i '1i #include <set>'   "$f"
  grep -q '#include <array>' "$f" || sed -i '1i #include <array>' "$f"
done
# 4. the replay writer emits ~114 MB per game; this sweep needs only result.json
python3 - <<'EOF'
import pathlib
p = pathlib.Path('bench/native_common.hpp'); s = p.read_text()
s = s.replace(
    'void capture(Engine& e,const J& actions=J::array(),bool force=false){',
    'bool replay_disabled=true;\n    void capture(Engine& e,const J& actions=J::array(),bool force=false){ if(replay_disabled) return;')
s = s.replace(
    'void save(Engine& e,const fs::path& p,const std::string& reason){capture(e,J::array(),true);data["summary"]=',
    'void save(Engine& e,const fs::path& p,const std::string& reason){ if(replay_disabled){ J s2; s2["summary"]={{"duration",e.time},{"score",e.score},{"frames",0},{"reason",reason}}; save_json(p,s2); return; } capture(e,J::array(),true);data["summary"]=')
p.write_text(s)
EOF

# deps. NumPy is pinned: the engine routes sin/cos/arctan2/hypot through NumPy's own
# compiled loops, and the seed-aware README says 1.26 results must not be pooled with 2.3.5.
apt-get update -qq >/dev/null 2>&1
DEBIAN_FRONTEND=noninteractive apt-get install -y -qq pkg-config nlohmann-json3-dev libcpp-httplib-dev >/dev/null 2>&1
pip install -q 'numpy==2.3.5' >/dev/null 2>&1
python3 build.py >/dev/null 2>&1

test -x bench/policy_bench || { echo BUILD_FAILED; exit 1; }

cat > /root/run_arm.sh <<'EOF'
#!/bin/bash
MODE=$1; SEED=$2; OUT=/root/w/$MODE-$SEED
cd /root/survival-simulator/seed-aware-policy
./bench/policy_bench configs/orchard.json $SEED $MODE 3000 $OUT 180 >/dev/null 2>&1
if [ -f $OUT/result.json ]; then
  python3 -c "
import json
d=json.load(open('$OUT/result.json'))
print(','.join(str(x) for x in ['$MODE','$SEED',d.get('score',''),d.get('duration',''),d.get('peak_population',''),sum(d.get('deaths_by_cause',[]))]))
" >> /root/res/$MODE.csv
fi
rm -rf $OUT
EOF
chmod +x /root/run_arm.sh
mkdir -p /root/w /root/res
echo BOOTSTRAP_OK
python3 -c "import numpy;print('numpy',numpy.__version__)"
nproc
