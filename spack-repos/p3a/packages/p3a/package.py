from spack import *


class P3a(CMakePackage):
    """This is a C++17 library that is meant to support High Performance Computing physics applications."""

    homepage = "https://github.com/sandialabs/p3a"
    git = "https://github.com/sandialabs/p3a.git"

    version("main", branch="main")

    depends_on("mpicpp")
    depends_on("kokkos")

    def cmake_args(self):
        args = ["-Dmpicpp_ROOT={}".format(self.spec["mpicpp"].prefix),
                "-Dkokkos_ROOT={}".format(self.spec["kokkos"].prefix)]

        if self.spec.satisfies("^kokkos+rocm"):
            args.append("-DCMAKE_CXX_COMPILER={}".format(self.spec["hip"].hipcc))
        else:
            # Compiler weirdness due to nvcc_wrapper
            args.append("-DCMAKE_CXX_COMPILER={}" .format(self.spec["kokkos"].kokkos_cxx))
        return args
