/*********************                                                        */
/*! \file Debug.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __Debug_h__
#define __Debug_h__

#include <cstdlib>
#include <stdio.h>


#ifndef NDEBUG
#define DEBUG( x ) x
#else
#define DEBUG( x )
#endif

#ifdef LOG
#undef LOG
#endif

#ifndef NDEBUG
#define LOG( x, f, y, ... )                                                                        \
    {                                                                                              \
        if ( ( x ) )                                                                               \
        {                                                                                          \
            printf( f, y );                                                                        \
        }                                                                                          \
    }
#else
#define LOG( x, f, y, ... )                                                                        \
    {                                                                                              \
    }
#endif

#ifndef NDEBUG
#define ASSERTM( x, y, ... )                                                                       \
    {                                                                                              \
        if ( !( x ) )                                                                              \
        {                                                                                          \
            printf( y );                                                                           \
            exit( 1 );                                                                             \
        }                                                                                          \
    }
#else
#define ASSERTM( x, y, ... )
#endif

#ifndef NDEBUG
#define ASSERT( x )                                                                                \
    {                                                                                              \
        if ( !( x ) )                                                                              \
        {                                                                                          \
            printf( "Assertion violation! File %s, line %d\n", __FILE__, __LINE__ );               \
            exit( 1 );                                                                             \
        }                                                                                          \
    }
#else
#define ASSERT( x )
#endif

#endif // __Debug_h__
