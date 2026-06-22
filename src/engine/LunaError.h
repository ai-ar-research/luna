/*********************                                                        */
/*! \file LunaError.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __LunaError_h__
#define __LunaError_h__

#include "Error.h"

class LunaError : public Error
{
public:
    enum Code {
        UNEXPECTED_TENSOR_SHAPE = 0,
        INVALID_NODE_TYPE = 1,
        INVALID_BOUND_COMPUTATION = 2,
        ALPHA_OPTIMIZATION_FAILED = 3,

        ONNX_PARSING_ERROR = 4,
        UNSUPPORTED_OPERATION = 5,
        INVALID_MODEL_STRUCTURE = 6,

        IBP_COMPUTATION_FAILED = 7,
        CROWN_BACKWARD_FAILED = 8,
        CONCRETE_BOUNDS_FAILED = 9,

        NODE_NOT_FOUND = 10,
        INVALID_NODE_INDEX = 11,
        UNINITIALIZED_NODE = 12,

        INTERNAL_ERROR = 99
    };

    LunaError( LunaError::Code code )
        : Error( "LunaError", (int)code )
    {
    }

    LunaError( LunaError::Code code, const char *userMessage )
        : Error( "LunaError", (int)code, userMessage )
    {
    }
};

#endif // __LunaError_h__
