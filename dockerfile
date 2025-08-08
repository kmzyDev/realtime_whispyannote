#
# BUILD BASE IMAGE (USED ONLY Windows11)
# ------------------------------------------------------------------------------
FROM debian:bullseye

# install dependent packages
RUN apt-get -y update && apt-get install -y --no-install-recommends \
    mingw-w64 \
    g++-mingw-w64-x86-64-posix \
    cmake \
    make \
    && update-alternatives --set x86_64-w64-mingw32-g++ /usr/bin/x86_64-w64-mingw32-g++-posix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# specify working directory
WORKDIR /workspace

CMD ["tail", "-f", "/dev/null"]
