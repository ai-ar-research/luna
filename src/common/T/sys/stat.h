/*********************                                                        */
/*! \file stat.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __T__sys__stat_h__
#define __T__sys__stat_h__

#include <sys/stat.h>

namespace T {
    inline int stat(const char *pathname, struct ::stat *statbuf) {
        return ::stat(pathname, statbuf);
    }
}

#endif // __T__sys__stat_h__
