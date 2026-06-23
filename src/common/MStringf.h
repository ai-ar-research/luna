/*********************                                                        */
/*! \file MStringf.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __Stringf_h__
#define __Stringf_h__

#include "MString.h"

#include <cstdarg>
#include <cstdio>

class Stringf : public String
{
public:
    enum {
        MAX_STRING_LENGTH = 10000,
    };

    Stringf( const char *format, ... )
    {
        va_list argList;
        va_start( argList, format );

        char buffer[MAX_STRING_LENGTH];

        vsnprintf( buffer, MAX_STRING_LENGTH, format, argList );

        va_end( argList );

        _super = Super( buffer );
    }
};

#endif // __Stringf_h__
