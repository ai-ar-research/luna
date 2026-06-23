/*********************                                                        */
/*! \file InputParserError.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __InputParserError_h__
#define __InputParserError_h__

#include "Error.h"

class InputParserError : public Error
{
public:
    enum Code {
        VARIABLE_INDEX_OUT_OF_RANGE = 0,
        UNEXPECTED_INPUT = 1,
        FILE_DOESNT_EXIST = 2,
        UNSUPPORTED_BOUND_TYPE = 3,
        NETWORK_LEVEL_REASONING_DISABLED = 4,
        HIDDEN_VARIABLE_DOESNT_EXIST_IN_NLR = 5,
    };

    InputParserError( InputParserError::Code code )
        : Error( "InputParserError", (int)code )
    {
    }

    InputParserError( InputParserError::Code code, const char *userMessage )
        : Error( "InputParserError", (int)code, userMessage )
    {
    }
};

#endif // __InputParserError_h__
