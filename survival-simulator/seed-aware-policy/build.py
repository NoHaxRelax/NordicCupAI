from pathlib import Path
import subprocess,sysconfig,shlex,numpy
here=Path(__file__).resolve().parent/'bench'
flags=['g++','-O3','-std=c++17','-march=native','-ffp-contract=off','-pthread']
http=shlex.split(subprocess.check_output(['pkg-config','--cflags','--libs','cpp-httplib'],text=True))
subprocess.run(flags+['-I'+sysconfig.get_path('include'),'-I'+numpy.get_include(),str(here/'policy_bench.cpp'),'-o',str(here/'policy_bench'),'-L'+sysconfig.get_config_var('LIBDIR'),'-lpython3.12','-ldl','-lm']+http,check=True)
