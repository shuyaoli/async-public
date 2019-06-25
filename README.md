# Compile
Use ```mex CXXFLAGS="\$CXXFLAGS -std=c++17"``` to compile *.cpp files from MATLAB, e.g.

```mex CXXFLAGS="\$CXXFLAGS -std=c++17" Finito_multi_threaded.cpp``` compiles Finito_multi_threaded.cpp

# Run MATLAB
Use ```LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 matlab &``` to run MATLAB from command line.

# Filename explained
Folder _cpp/_ is used to store testing .cpp files; it's unimportant.

_main\_*.m_ with "MATLAB" in * is implemented with MATLAB.

_main\_*.m_ with "mex" in * is implemented with mex and C++.

All files ending with "direct" means the implementation is straightforward from Finito paper without further optimization, i.e. different from the one in the textbook.

Finito\_single\_threaded.cpp is the final version for single threaded implementation.

**Finito_multi_threaded.cpp is the final version for multi threaded implementation. The associated main.m file is main_mex_multi_threaded**

\_no\_CAS means the implementation **locks mean\_z rather than use _compare and swap_**; it's speed is roughly the same as the wait-free version.
