/*********************                                                        */
/*! \file Errno.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __T__Errno_h__
#define __T__Errno_h__

#include <cerrno>

namespace T {
    inline int errorNumber() {
        return errno;
    }
}

#endif // __T__Errno_h__
