/*********************                                                        */
/*! \file stdlib.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __T__Stdlib_h__
#define __T__Stdlib_h__

#include <cstdlib>

namespace T {
    inline void* malloc(size_t size) {
        return ::malloc(size);
    }

    inline void free(void* ptr) {
        ::free(ptr);
    }

    inline void* realloc(void* ptr, size_t size) {
        return ::realloc(ptr, size);
    }

    inline void srand(unsigned seed) {
        ::srand(seed);
    }

    inline int rand() {
        return ::rand();
    }
}

#endif // __T__Stdlib_h__
