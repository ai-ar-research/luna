/*********************                                                        */
/*! \file TensorUtils.h
 ** \verbatim
 ** This file is part of the Luna project.
 ** Copyright (c) 2025-2026 by the authors listed in the file AUTHORS
 ** in the top-level source directory) and their institutional affiliations.
 ** All rights reserved. See the file COPYING in the top-level source
 ** directory for licensing information.\endverbatim
 **
 **/

#ifndef __TensorUtils_h__
#define __TensorUtils_h__

#include "Debug.h"
#include "MString.h"
#include "Vector.h"

typedef Vector<unsigned int> TensorShape;
typedef unsigned int TensorIndex;
typedef int SignedTensorIndex;
typedef Vector<TensorIndex> TensorIndices;
typedef TensorIndex PackedTensorIndices;

typedef Vector<unsigned int> Permutation;

TensorIndices unpackIndex( TensorShape shape, PackedTensorIndices packedIndex );

PackedTensorIndices packIndex( TensorShape shape, TensorIndices indices );

unsigned int tensorSize( TensorShape shape );

template <typename T> T tensorLookup( Vector<T> tensor, TensorShape shape, TensorIndices indices )
{
    return tensor[packIndex( shape, indices )];
}

template <typename T> Vector<T> transposeVector( Vector<T> values, Permutation permutation )
{
    Vector<T> result;
    for ( unsigned int i : permutation )
    {
        ASSERT( i < values.size() );
        result.append( values[i] );
    }
    return result;
}

template <typename T>
Vector<T> transposeTensor( Vector<T> tensor, TensorShape shape, Permutation permutation )
{
    ASSERT( shape.size() == permutation.size() );
    ASSERT( tensorSize( shape ) == tensor.size() );

    TensorShape transposedShape = transposeVector( shape, permutation );
    Vector<T> result( tensor.size() );
    for ( PackedTensorIndices rawInputIndex = 0; rawInputIndex < tensor.size(); rawInputIndex++ )
    {
        TensorIndices inputIndex = unpackIndex( shape, rawInputIndex );
        TensorIndices outputIndex = transposeVector( inputIndex, permutation );
        int rawOutputIndex = packIndex( transposedShape, outputIndex );
        result[rawOutputIndex] = tensor[rawInputIndex];
    }
    return result;
}

TensorShape getMultidirectionalBroadcastShape( TensorShape shape1, TensorShape shape2 );

TensorIndices broadcastIndex( TensorShape currentShape,
                              TensorShape broadcastShape,
                              TensorIndices broadcastIndices );

TensorIndex unsignIndex( unsigned int size, SignedTensorIndex signedIndex );

Permutation reversePermutation( unsigned int size );

struct Padding
{
public:
    int padFront;
    int padBack;

    Padding( int padFront, int padBack );
};

Padding
calculatePaddingNeeded( int inputSize, int filterSize, int stride, bool padFrontPreferentially );

#endif // __TensorUtils_h__