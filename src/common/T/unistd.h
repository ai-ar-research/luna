/*********************                                                        */
/*! \file unistd.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __T__unistd_h__
#define __T__unistd_h__

#include <unistd.h>
#include <fcntl.h>

namespace T {
    inline int open(const char *pathname, int flags) {
        return ::open(pathname, flags);
    }

    inline int open(const char *pathname, int flags, mode_t mode) {
        return ::open(pathname, flags, mode);
    }

    inline ssize_t read(int fd, void *buf, size_t count) {
        return ::read(fd, buf, count);
    }

    inline ssize_t write(int fd, const void *buf, size_t count) {
        return ::write(fd, buf, count);
    }

    inline int close(int fd) {
        return ::close(fd);
    }
}

#endif // __T__unistd_h__
