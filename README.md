bash ./scripts/configure.sh
bash ./scripts/build.sh
bash ./scripts/run.sh
bash ./scripts/clang-format-all.sh ./src ./apps

Todo
1. Use attached meshes!
2. Refactoring HIP kernals (camelCase naming, remove gpustructs.hpp)
3. Getters/Setters
4. Public Api cleaning
5. Check memory leaks
6. pixel artifacts