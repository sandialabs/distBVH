from spack import *


class Mpicpp(CMakePackage):
    """MPICPP is a simple C++ interface for the MPI C library standard. It brings a few tried-and-true C++ principles to bear over the plain C API:"""

    homepage = "https://github.com/sandialabs/mpicpp"
    git = "https://github.com/sandialabs/mpicpp.git"

    version("main", branch="main")

    depends_on("mpi")

    def cmake_args(self):
        args = ["-DMPI_HOME={}".format(self.spec["mpi"].prefix)]
        return args
